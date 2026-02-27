from nrc_web_teleop_helpers.stretch_ik_control import StretchIKControl
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from nrc_web_teleop_helpers.constants import Joint, Frame

# 사용할 pinocchio 모델 생성
"""
hello-robot-stretch-urdf 패키지 필요
설치 방법 (https://github.com/hello-robot/stretch_urdf)
"""
import stretch_urdf.urdf_utils as uu

# 기본 URDF를 가공하여 app을 위한 임시 URDF 생성하고, 그 경로를 반환
urdf_fpaths = uu.generate_ik_urdfs("nrc_web_teleop", rigid_wrist_urdf=False)
"""
    generate_ik_urdfs(app_name, rigid_wrist_urdf=True, base_rotation=True)
    
    Generates URDFs for IK packages. The latest calibrated
    URDF is used as a starting point, then these modifications
    are applied:
      1. Clip joint limits
      2. Make non-IK joints rigid
      3. Merge arm joints환
      4. Add virtual rotary base joint
      5. (optionally) Make wrist joints rigid

    Parameters
    ----------
    app_name : str
        the name of your application
    rigid_wrist_urdf : bool or None
        whether to also generate a IK URDF with a fixed dex wrist

    Returns
    -------
    list(str)
        one or two filepaths, depending on `rigid_wrist_urdf`,
        to the generated URDFs. The first element will be the
        full IK version, and the second will be the rigid
        wrist version.
    
"""
urdf_fpath = urdf_fpaths[0] # full IK version의 URDF 경로
# urdf_fpath = urdf_fpaths[1] # rigid wrist version의 URDF 경로

# URDF 파일으로 pinocchio 모델 생성
model = pin.buildModelFromUrdf(urdf_fpath)
data = model.createData()

# nq는 관절 각도 수. 즉, 모델의 관절 자유도. 0번은 항상 universe joint (고정된 조인트)임.
print("=" * 40)
print("All joint names in the model:")
for i in range(model.nq+1):
    print(f"{i}: {model.names[i]}")
print("=" * 40)
print("All frame names in the model:")
for i in range(model.nframes):
    print(f"{i}: {model.frames[i].name}")

# base link 프레임을 기준으로 한 프레임 변환 계산
# CompressedImage 객체는 header.frame_id에서 수집된 Image의 프레임 id를 포함.
base_link_frame_id = Frame.BASE_LINK.value

# get_goal_pose (goal pose in camera frame을 base link frame으로 변환)
# pose_transformed = tf_buffer.transform(pose, target_frame, timeout)