# Third-party Source Directory

This directory is reserved for third-party source code used by the reproduction workflow. The third-party repositories themselves are not committed here.

Expected GSNet / Graspness source:

```bash
git clone https://github.com/graspnet/graspness_unofficial.git external/graspness_unofficial
```

Reference upstream commit:

```text
b5abf5aaaf3a797514f89161d8ccc8dfc2ec0eca
```

GraspNet evaluation also requires `graspnetAPI` and the compiled `grasp_nms` extension. Install them in the active environment, for example:

```bash
git clone --depth 1 https://github.com/graspnet/graspnetAPI.git external/graspnetAPI
conda run -n gsnet-repro-sm120 pip install -e external/graspnetAPI
```

`external/graspness_unofficial/` and `external/graspnetAPI/` are ignored by `.gitignore`.
