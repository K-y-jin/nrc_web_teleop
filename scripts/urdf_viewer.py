"""
output/stretch.urdf 파일을 MeshcatVisualizer로 시각화하는 뷰어 스크립트.

실행:
    python3 scripts/urdf_viewer.py [urdf_path]

urdf_path를 생략하면 output/stretch.urdf를 사용한다.
실행 후 출력되는 localhost URL을 브라우저로 열면 로봇 모델을 볼 수 있다.

주의: URDF가 `./meshes/...` 상대 경로를 쓰지만 output/ 폴더에는 meshes/ 가
없으므로, stretch_urdf 패키지의 메쉬 디렉토리 중 모든 참조를 만족하는 것을
자동으로 찾아 URDF의 경로를 절대 경로로 치환한 임시 URDF를 만들어 로드한다.
"""
import os
import re
import sys
import tempfile
import time

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_URDF = os.path.join(PROJECT_ROOT, "output", "stretch.urdf")

MESH_ATTR_RE = re.compile(r'(<mesh\b[^>]*\bfilename=")([^"]+)(")')


def collect_mesh_candidate_dirs():
    """stretch_urdf 패키지와 알려진 경로에서 `meshes/`를 포함한 디렉토리를 수집."""
    candidates = []
    try:
        import stretch_urdf  # noqa: F401
        stretch_urdf_root = os.path.dirname(stretch_urdf.__file__)
        for variant in ("SE3", "RE2V0", "RE1V0"):
            variant_dir = os.path.join(stretch_urdf_root, variant)
            if os.path.isdir(os.path.join(variant_dir, "meshes")):
                candidates.append(variant_dir)
    except ImportError:
        pass

    extra_roots = [
        "/home/nrc/Documents/stretch/stretch_urdf/stretch_urdf",
        "/home/nrc/Documents/stretch/stretch_ros2/stretch_description",
    ]
    for root in extra_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if "meshes" in dirnames:
                candidates.append(dirpath)

    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def extract_mesh_refs(urdf_text):
    """URDF에서 <mesh filename="..."/> 의 파일명 목록을 추출."""
    return [m.group(2) for m in MESH_ATTR_RE.finditer(urdf_text)]


def resolve_mesh_root(urdf_path, mesh_refs):
    """URDF가 참조하는 모든 메쉬를 담고 있는 디렉토리를 선택.

    반환 값은 상대 경로 `<filename>`을 붙여 실제 파일 경로가 되는 base dir.
    예) mesh_refs = ['./meshes/base_link.STL', ...] 라면
        base dir 안에 'meshes/base_link.STL' 이 존재해야 한다.
    """
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    def dir_has_all(base):
        for ref in mesh_refs:
            p = os.path.normpath(os.path.join(base, ref))
            if not os.path.isfile(p):
                return False
        return True

    # 1) URDF 같은 디렉토리에 이미 meshes가 있는 경우
    if dir_has_all(urdf_dir):
        return urdf_dir

    # 2) stretch_urdf 후보 디렉토리 순회
    for cand in collect_mesh_candidate_dirs():
        if dir_has_all(cand):
            return cand
    return None


def rewrite_urdf_with_absolute_meshes(urdf_path, mesh_base_dir):
    """URDF 파일의 mesh filename을 절대 경로로 치환한 임시 URDF를 생성."""
    with open(urdf_path, "r") as f:
        text = f.read()

    def _sub(match):
        prefix, filename, suffix = match.group(1), match.group(2), match.group(3)
        abs_path = os.path.normpath(os.path.join(mesh_base_dir, filename))
        return f"{prefix}{abs_path}{suffix}"

    new_text = MESH_ATTR_RE.sub(_sub, text)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=False, prefix="stretch_viewer_"
    )
    tmp.write(new_text)
    tmp.close()
    return tmp.name


def main():
    urdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URDF
    if not os.path.isfile(urdf_path):
        print(f"URDF not found: {urdf_path}")
        sys.exit(1)

    with open(urdf_path, "r") as f:
        urdf_text = f.read()
    mesh_refs = extract_mesh_refs(urdf_text)
    print(f"URDF: {urdf_path}")
    print(f"Mesh references: {len(mesh_refs)}")

    effective_urdf = urdf_path
    mesh_base = resolve_mesh_root(urdf_path, mesh_refs)
    if mesh_base is None:
        print("WARNING: 모든 메쉬를 포함한 디렉토리를 찾지 못했습니다. "
              "모델이 표시되지 않을 수 있습니다.")
    else:
        print(f"Mesh base dir: {mesh_base}")
        effective_urdf = rewrite_urdf_with_absolute_meshes(urdf_path, mesh_base)
        print(f"Rewritten URDF: {effective_urdf}")

    # Pinocchio 모델 및 시각화 모델 로드
    model = pin.buildModelFromUrdf(effective_urdf)
    visual_model = pin.buildGeomFromUrdf(
        model, effective_urdf, pin.GeometryType.VISUAL
    )
    collision_model = pin.buildGeomFromUrdf(
        model, effective_urdf, pin.GeometryType.COLLISION
    )

    print("=" * 40)
    print(f"Model  nq={model.nq}, nv={model.nv}, nframes={model.nframes}")
    print(f"Visual geoms   : {len(visual_model.geometryObjects)}")
    print(f"Collision geoms: {len(collision_model.geometryObjects)}")
    print("Joints:")
    for i in range(model.njoints):
        print(f"  {i}: {model.names[i]}")

    if len(visual_model.geometryObjects) == 0:
        print("WARNING: visual geometry가 0개입니다. 메쉬 로드에 실패한 것 같습니다.")

    # MeshcatVisualizer로 표시
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=False)
    viz.loadViewerModel()

    q = pin.neutral(model)
    viz.display(q)

    # 카메라를 로봇이 보이는 위치로 이동 (stretch는 약 1.5m 높이)
    try:
        import meshcat.transformations as tf
        cam_tf = tf.translation_matrix([1.0, 1.0, 0.6])
        viz.viewer["/Cameras/default"].set_transform(cam_tf)
        # 줌 배율 (값이 클수록 확대)
        viz.viewer["/Cameras/default/rotated/<object>"].set_property("zoom", 2.0)
    except Exception as e:
        print(f"Camera setup failed: {e}")

    viewer_url = viz.viewer.url()
    print("=" * 40)
    print(f"Meshcat viewer: {viewer_url}")
    print("Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
