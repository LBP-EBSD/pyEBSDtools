"""
EMsoft runner — Stage 2 of the data generation pipeline.

Responsibilities:
  1. NMLWriter  — generates all four NML config files from a config dict.
  2. DockerRunner — manages the Docker image and runs the 3 EMsoft commands
                    (EMMCOpenCL → EMEBSDmaster[OpenCL] → EMEBSD) inside a container.

Docker image strategy (set docker.image in config.yaml):
  - Pre-built (Docker Hub): marcdegraef/emsoft_sdk:buildx-latest
  - Local build:            emsoft:local  (built via `make docker-build`)

Volume mounts inside the container:
  - /home/EMuser/EMPlay      ← paths.data_dir  (experiment outputs live here)
  - /home/EMuser/XtalFolder  ← paths.xtal_dir  (crystal .xtal files live here)

All NML file paths are relative to EMdatapathname (/home/EMuser/EMPlay).
"""

import os
import subprocess
import textwrap
import shlex
from pathlib import Path


# ─── Container-side paths (fixed by EMsoft Docker image) ─────────────────────
_CONTAINER_EMPLAY  = "/home/EMuser/EMPlay"
_CONTAINER_XTAL    = "/home/EMuser/XtalFolder"
_CONTAINER_CONFIG  = "/root/.config/EMsoft/EMsoftConfig.json"


class NMLWriter:
    """
    Writes the four NML files required by the EMsoft EBSD pipeline.

    All paths inside NML files are relative to EMdatapathname.
    """

    def __init__(self, cfg: dict):
        self.cfg       = cfg
        self.ems       = cfg["emsoft"]
        self.paths     = cfg["paths"]
        self.exp_name  = self.paths["experiment_name"]

    def write_all(self, host_exp_dir: str) -> dict[str, str]:
        """
        Write all NML files to `host_exp_dir` on the host.

        Returns:
            dict mapping role → absolute host path.
        """
        os.makedirs(host_exp_dir, exist_ok=True)
        written = {}
        for role, (fname, content) in self._all_nmls().items():
            path = os.path.join(host_exp_dir, fname)
            with open(path, "w") as fh:
                fh.write(content)
            written[role] = path
            print(f"[nml] Wrote {role:<16} → {path}")
        return written

    # ─── NML content generators ───────────────────────────────────────────────

    def _all_nmls(self) -> dict[str, tuple[str, str]]:
        """Return {role: (filename, content)} for every NML."""
        exp = self.exp_name
        return {
            "mc":     (f"EMMCOpenCL.nml",         self._mc_nml()),
            "master": (f"EMEBSDmaster.nml",        self._master_nml()),
            "master_gpu": (f"EMEBSDmasterOCL.nml", self._master_gpu_nml()),
            "bethe":  (f"BetheParameters.nml",     self._bethe_nml()),
            "ebsd":   (f"EMEBSD.nml",              self._ebsd_nml()),
        }

    def _mc_nml(self) -> str:
        ems = self.ems
        exp = self.exp_name
        return textwrap.dedent(f"""\
             &MCCLdata
              mode = 'full',
              xtalname = '{ems['xtalname']}',
              sig = {ems['sample_tilt_deg']},
              omega = 0.0,
              numsx = {ems.get('mc_grid_size', 801)},
              num_el = 10,
              globalworkgrpsz = 150,
              totnum_el = {ems['n_mc_electrons']},
              multiplier = 1,
              EkeV = {ems['accelerating_voltage_keV']},
              Ehistmin = 15.0,
              Ebinsize = 1.0,
              depthmax = 100.0,
              depthstep = 1.0,
              platid = 1,
              devid = 1,
              dataname = '{exp}/Fe_MCoutput.h5',
              Notify = 'Off',
             /
        """)

    def _master_nml(self) -> str:
        ems = self.ems
        exp = self.exp_name
        return textwrap.dedent(f"""\
             &EBSDmastervars
              dmin = 0.05,
              npx = 500,
              nthreads = {ems['n_threads']},
              doLegendre = .FALSE.,
              energyfile = '{exp}/Fe_MCoutput.h5',
              BetheParametersFile = '{exp}/BetheParameters.nml',
              Notify = 'Off',
             /
        """)

    def _master_gpu_nml(self) -> str:
        ems = self.ems
        exp = self.exp_name
        # nthreads must be 4N+3 minimum 7 for OpenCL master
        nthreads = max(7, ems['n_threads'])
        if (nthreads - 3) % 4 != 0:
            nthreads = ((nthreads - 3) // 4 + 1) * 4 + 3
        return textwrap.dedent(f"""\
             &EBSDmastervars
              dmin = 0.05,
              npx = 500,
              nthreads = {nthreads},
              platid = 1,
              devid = 1,
              globalworkgrpsz = 150,
              blocksize = 32,
              energyfile = '{exp}/Fe_MCoutput.h5',
              BetheParametersFile = '{exp}/BetheParameters.nml',
              restart = .FALSE.,
              uniform = .FALSE.,
             /
        """)

    def _bethe_nml(self) -> str:
        return textwrap.dedent("""\
             &BetheList
              c1 = 8.0,
              c2 = 50.0,
              c3 = 100.0,
              sgdbdiff = 1.0,
             /
        """)

    def _ebsd_nml(self) -> str:
        ems  = self.ems
        exp  = self.exp_name
        gen  = self.cfg["generation"]
        angles_file = f"{exp}/{exp}_angles.txt"
        master_file = f"{exp}/Fe_MCoutput.h5"   # master appends into the MC file
        data_file   = f"{exp}/Fe_EBSD_patterns.h5"
        return textwrap.dedent(f"""\
             &EBSDdata
              L = {ems['camera_distance_um']},
              thetac = {ems['detector_tilt_deg']},
              delta = 50.0,
              numsx = {ems['pattern_width']},
              numsy = {ems['pattern_height']},
              xpc = 0.0,
              ypc = 0.0,
              energymin = {ems['energy_min_keV']},
              energymax = {ems['energy_max_keV']},
              includebackground = 'n',
              anglefile = '{angles_file}',
              anglefiletype = 'orpcdef',
              eulerconvention = 'tsl',
              masterfile = '{master_file}',
              datafile = '{data_file}',
              bitdepth = 'float',
              beamcurrent = 150.0,
              dwelltime = 100.0,
              poisson = 'n',
              binning = 1,
              applyDeformation = 'y',
              Fframe = 'crys',
              scalingmode = 'not',
              gammavalue = 1.0,
              makedictionary = 'n',
              maskpattern = 'n',
              nthreads = {ems['n_threads']},
             /
        """)


class DockerRunner:
    """
    Manages the EMsoft Docker container and runs the 3-step EBSD pipeline.

    Steps executed inside the container (sequentially):
        1. EMMCOpenCL      — Monte Carlo electron trajectories
        2. EMEBSDmasterOpenCL / EMEBSDmaster — master diffraction pattern
        3. EMEBSD          — synthetic pattern stack with deformation

    Args:
        cfg: Full config dict (from config.yaml).
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.docker   = cfg["docker"]
        self.paths    = cfg["paths"]
        self.ems      = cfg["emsoft"]
        self.exp_name = self.paths["experiment_name"]

        self.host_data_dir = os.path.expanduser(self.paths["data_dir"])   # ~/EMsoftData
        self.host_xtal_dir = os.path.expanduser(self.paths["xtal_dir"])   # ~/EMsoftData/XtalFolder
        self.image         = self.docker["image"]

    def ensure_image(self) -> None:
        """Pull the Docker image if it is not available locally."""
        result = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"[docker] Image '{self.image}' found locally.")
            return

        print(f"[docker] Image '{self.image}' not found — pulling...")
        self._run_cmd(["docker", "pull", self.image], stream=True)

    def write_container_config(self) -> None:
        """
        No-op: the marcdegraef/emsoft:buildx-latest image ships with a correct
        EMsoftConfig.json pre-baked at /home/EMuser/.config/EMsoft/ that already
        points EMdatapathname → /home/EMuser/EMPlay and
               EMXtalFolderpathname → /home/EMuser/XtalFolder.
        Both of those map exactly to our volume mounts, so no override is needed.
        """
        print("[docker] Using pre-baked EMsoftConfig.json from container image.")

    def run_pipeline(self) -> None:
        """
        Run the full 3-step EMsoft pipeline.

        EMMCOpenCL always uses GPU/OpenCL. When use_gpu=True, EMEBSDmasterOpenCL
        is used (fast GPU master). Because the GPU master writes a different HDF5
        group name ('EBSDMasterOpenCLNameList') than what EMEBSD reads
        ('EBSDMasterNameList'), we patch the file on the host with h5py between
        the master and EMEBSD steps.

        Pipeline:
          Docker run 1: EMMCOpenCL + EMEBSDmaster[OpenCL]
          Host step   : h5py group rename (GPU master only)
          Docker run 2: EMEBSD
        """
        exp = self.exp_name
        use_gpu = self.ems.get("use_gpu", True)
        xtalname = self.ems.get("xtalname", "Fe_FCC.xtal")
        mc_out = os.path.join(self.host_data_dir, exp, "Fe_MCoutput.h5")
        patterns_out = os.path.join(self.host_data_dir, exp, "Fe_EBSD_patterns.h5")

        master_cmd = (
            f"EMEBSDmasterOpenCL {exp}/EMEBSDmasterOCL.nml"
            if use_gpu else
            f"EMEBSDmaster {exp}/EMEBSDmaster.nml"
        )

        # GPU flags — always required because EMMCOpenCL always uses OpenCL.
        gpu_flags = self._gpu_flags()

        # Mount the host's OpenCL ICD vendors so the container can find the
        # NVIDIA OpenCL platform.
        opencl_mounts = []
        if os.path.isdir("/etc/OpenCL/vendors"):
            opencl_mounts = ["-v", "/etc/OpenCL/vendors:/etc/OpenCL/vendors:ro"]

        xtal_setup = (
            f"XTAL_SRC=$(find /home/EMs/EMsoftData -name 'Ni.xtal' 2>/dev/null | head -1) && "
            f"if [ -z \"$XTAL_SRC\" ]; then "
            f"  echo '[emsoft] ERROR: Ni.xtal not found inside container' && exit 1; "
            f"fi && "
            f"mkdir -p /tmp/XtalFolder && "
            f"cp \"$XTAL_SRC\" /tmp/XtalFolder/{xtalname} && "
            f"python3 -c \""
            f"import json; cfg='/home/EMuser/.config/EMsoft/EMsoftConfig.json'; "
            f"c=json.load(open(cfg)); c['EMXtalFolderpathname']='/tmp/XtalFolder'; "
            f"json.dump(c, open(cfg,'w'), indent=4)"
            f"\" && "
            f"echo '[emsoft] Xtal ready: /tmp/XtalFolder/{xtalname}' && "
        )

        def _docker_cmd(bash: str) -> list[str]:
            return [
                "docker", "run", "--rm",
                *gpu_flags,
                *opencl_mounts,
                "-v", f"{self.host_data_dir}:{_CONTAINER_EMPLAY}",
                self.image,
                "bash", "-c", bash,
            ]

        print(f"[docker] Running EMsoft pipeline (gpu={use_gpu})...")
        print(f"[docker] Image : {self.image}")
        print(f"[docker] Data  : {self.host_data_dir} → {_CONTAINER_EMPLAY}")
        if opencl_mounts:
            print(f"[docker] OpenCL: /etc/OpenCL/vendors mounted from host")
        print()

        # ── Docker run 1: MC + master ─────────────────────────────────────────
        mc_container = os.path.join(exp, "Fe_MCoutput.h5")
        script_mc_master = (
            f"set -e && "
            f"{xtal_setup}"
            f"cd /home/EMuser/EMPlay && "
            f"echo '[emsoft] Step 1: Monte Carlo...' && "
            f"EMMCOpenCL {exp}/EMMCOpenCL.nml ; "
            f"[ -f {mc_container} ] || {{ echo '[emsoft] FATAL: EMMCOpenCL produced no output'; exit 1; }} && "
            f"echo '[emsoft] Step 2: Master pattern...' && "
            f"{master_cmd}"
        )
        self._run_cmd(_docker_cmd(script_mc_master), stream=True)

        # ── Host step: patch HDF5 group name (GPU master only) ────────────────
        if use_gpu:
            self._patch_master_group(mc_out)

        # ── Docker run 2: EMEBSD pattern generation ───────────────────────────
        patterns_container = os.path.join(exp, "Fe_EBSD_patterns.h5")
        script_ebsd = (
            f"set -e && "
            f"{xtal_setup}"           # xtal needed here too (new container, clean /tmp)
            f"cd /home/EMuser/EMPlay && "
            f"echo '[emsoft] Step 3: Pattern generation...' && "
            f"EMEBSD {exp}/EMEBSD.nml ; "
            f"[ -f {patterns_container} ] || {{ echo '[emsoft] FATAL: EMEBSD produced no output'; exit 1; }} && "
            f"echo '[emsoft] Done.'"
        )
        self._run_cmd(_docker_cmd(script_ebsd), stream=True)

    def _patch_master_group(self, h5_path: str) -> None:
        """
        Copy NMLparameters/EBSDMasterOpenCLNameList →
              NMLparameters/EBSDMasterNameList
        inside Fe_MCoutput.h5 so EMEBSD can read GPU-master output.

        The file is owned by EMuser (uid=501, created inside Docker). The host
        user can't open it for writing, so we first chmod it via Docker (as root),
        then patch with h5py on the host, then restore sane permissions.
        """
        import h5py
        src_name = "EBSDMasterOpenCLNameList"
        dst_name = "EBSDMasterNameList"

        print(f"[h5py] Patching master group: {src_name} → {dst_name}")

        # The file is owned by EMuser inside the container. Use Docker as root
        # to make it world-writable so the host user can open it with h5py.
        container_path = h5_path.replace(self.host_data_dir, _CONTAINER_EMPLAY, 1)
        self._run_cmd([
            "docker", "run", "--rm", "--user", "root",
            "-v", f"{self.host_data_dir}:{_CONTAINER_EMPLAY}",
            self.image,
            "chmod", "666", container_path,
        ])

        with h5py.File(h5_path, "a") as f:
            nml = f["NMLparameters"]
            if dst_name in nml:
                print(f"[h5py] {dst_name} already present — skipping patch.")
                return
            if src_name not in nml:
                raise RuntimeError(
                    f"[h5py] Neither {src_name} nor {dst_name} found in "
                    f"NMLparameters — did the GPU master run succeed?"
                )
            nml.copy(src_name, nml, name=dst_name)
        print(f"[h5py] Patch complete.")

    # ─── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _gpu_flags() -> list[str]:
        """Return Docker GPU passthrough flags if an NVIDIA GPU is visible."""
        flags = ["--gpus", "all"]
        for dev in ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"]:
            if os.path.exists(dev):
                flags += ["--device", dev]
        return flags

    @staticmethod
    def _run_cmd(cmd: list[str], stream: bool = False) -> None:
        if stream:
            # Stream stdout/stderr live so the user sees progress
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                proc.wait()
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(proc.returncode, cmd)
        else:
            subprocess.run(cmd, check=True)


# ─── Convenience entry point ──────────────────────────────────────────────────

def run_from_config(cfg: dict, nml_paths: dict | None = None) -> None:
    """
    Write NML files and run the full EMsoft pipeline from a config dict.

    Args:
        cfg:       Parsed config.yaml dict.
        nml_paths: If provided, skip NML writing (already done).
    """
    paths    = cfg["paths"]
    data_dir = os.path.expanduser(paths["data_dir"])
    exp_dir  = os.path.join(data_dir, paths["experiment_name"])

    writer = NMLWriter(cfg)
    if nml_paths is None:
        writer.write_all(exp_dir)

    runner = DockerRunner(cfg)
    runner.ensure_image()
    runner.write_container_config()
    runner.run_pipeline()
