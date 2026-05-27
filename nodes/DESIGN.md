# 노드 설계 (nodes/)

ROS2 노드별 기능 요약. 빌드 등록은 `CMakeLists.txt` 참고.

> **개발 범위**: `move_base_to_point.py`, `move_gripper_to_point.py` 두 노드만
> 직접 개발/수정한다 (작성자 본인 코드). 그 외 노드는 외부 코드이므로 수정하지 않고
> 그대로 둔다.

## 카메라 / 영상 *(수정 대상 아님)*

| 노드 | 기능 |
|------|------|
| `gripper_camera.py` | 그리퍼 UVC 카메라(`/dev/hello-gripper-camera`)를 열어 `Image` 토픽으로 퍼블리시 |
| `navigation_camera.py` | 헤드 내비게이션 UVC 카메라(`/dev/hello-nav-head-camera`)를 `Image` 토픽으로 퍼블리시 |
| `old_navigation_camera.py` | 구버전 내비게이션 카메라 노드. 신규 노드로 대체됨 |
| `configure_video_streams.py` | RealSense·그리퍼·내비게이션 영상을 크롭/마스크/회전 가공하고, depth AR·신체 포즈 AR 오버레이를 입혀 압축 영상으로 재퍼블리시 |
| `compressed_image_visualizer.py` | ROS2 압축 영상을 CV2 창으로 띄워 보는 디버깅용 노드 |

## 모션 — 그래스프 *(수정 대상 아님)*

| 노드 | 기능 |
|------|------|
| `move_to_pregrasp.py` | 클릭한 물체에 대한 프리그래스프 자세로 팔을 이동시키는 `MoveToPregrasp` 액션 서버 |

## 음성 *(수정 대상 아님)*

| 노드 | 기능 |
|------|------|
| `text_to_speech.py` | `TextToSpeech` 토픽을 구독해 TTS 엔진(pyttsx3/gTTS)으로 음성 출력 |
| `text_to_speech_ui.py` | 터미널에서 텍스트를 입력해 TTS 명령을 보내는 대화형 CLI 노드 |

---

# 개발 대상 노드

아래 두 노드만 개발한다. 공통적으로 다음 구조를 따른다.

- `Node`를 상속한 액션 서버. 한 번에 하나의 goal만 처리(`active_goal_request`로 잠금).
- `__init__`에서 TF2·`StretchIKControl`(역자코비안 IK 컨트롤러)·카메라 구독 설정,
  `initialize()`에서 컨트롤러 초기화 후 액션 서버 생성.
- `goal_callback`에서 필요한 센서 메시지 수신 여부를 확인해 goal 수락/거부.
- `execute_callback`에서 클릭 좌표를 관절 목표값으로 변환하고,
  `MoveToActionState` 상태 머신을 5 Hz 루프로 순차 실행.
- 진행 상황은 `Feedback`(`new_scaled_u/v`, `elapsed_time`)으로 퍼블리시.

## `move_base_to_point.py` — 베이스 이동 *(재설계 — 목표 동작)*

내비게이션 카메라에서 클릭한 지점으로 로봇 **베이스**를 이동시킨다.
동작은 **(A) 거리 미리보기**(클릭 시)와 **(B) Move Base 액션**(버튼 클릭 시)
두 단계로 나뉜다.

### 신규 상수 (`constants.py`)
- `TILT_READY` — 거리 미리보기·베이스 이동에 쓰는 고정 head_tilt 각도(rad).
  이 각도에서 nav 카메라에 로봇 자신의 base가 보인다. **head_tilt만** 가리키며
  head_pan은 포함하지 않는다.
- `DEPTH_REF` — `TILT_READY` 자세에서 nav 카메라에 보이는 base 기준점을
  `ref_rcz` 형태 `(row, col, depth_mm)`로 둔 상수. `get_pred_depth`
  (Depth-Anything) 예측의 metric 스케일을 고정하는 기준 픽셀·기준 깊이를
  제공한다. (move_gripper_to_point의 RealSense 기반
  `align_realsense_ref_to_navigation`을 상수로 대체 → RealSense 구독 불필요.)
  추후 head_pan 값에 따라 가변적인 `DEPTH_REF`를 array 형태로 확장할 수 있다.

### (A) 거리 미리보기 — nav 카메라 뷰 클릭 시  *(요구사항 0)*

웹 UI(`CameraView.tsx`)에서 오버헤드(=내비게이션) 카메라 뷰를 클릭하면
`requestDistance` → `/get_distance` 서비스(`nrc_web_teleop/srv/GetDistance` —
요청 `scaled_u`/`scaled_v`, 응답 `distance`(m) / `success` /
`is_navigable` / `stop_scaled_u` / `stop_scaled_v`)가 호출된다. 이 서비스를
`move_base_to_point.py`가 제공한다.

1. 현재 `head_tilt`를 읽는다.
2. `head_tilt`가 `TILT_READY`가 **아니면** → head를 `TILT_READY`로 이동시키고
   `success=False`로 응답. 정렬 후 다시 클릭해야 거리가 나온다.
3. `head_tilt`가 `TILT_READY`이면 → 클릭 좌표까지의 거리 계산:
   - nav RGB → `get_pred_depth(rgb, ref_rcz=DEPTH_REF)` (내부적으로 mm)
     → `compute_click_point_in_base` 에서 m 로 변환.
   - 클릭 픽셀의 정규화 ray × depth → 광학 프레임의 3D 점(m).
   - `NAV_OPTICAL_FROM_LINK` 와 `tf_lookup_matrix(BASE_LINK, NAV_CAM_LINK)`
     로 base_link 프레임 3D 점(m)으로 변환.
   - 마커-base 수평거리 `distance = hypot(cx, cy)` (m).
     **DEFAULT_ARM_LENGTH는 미리보기에서 차감하지 않는다** (raw 값을 그대로
     `distance`로 반환; UI green 라벨 표기용).
   - `stop_scaled_u`/`stop_scaled_v` — `DEFAULT_ARM_LENGTH` 만큼 base 쪽으로
     당긴 base 정지 예상 위치를 nav 이미지상의 DEPTH_REF↔클릭 직선상에 비례
     투영한 정규화 픽셀 좌표. UI red 마커 표기용.
   - `is_navigable` — 일단 사용하지 않음(항상 `True`로 응답). 추후
     `is_line_navigable(pred_depth, py, px)`로 활성화 예정 (함수는 보존).

**overlay 렌더링**: `CameraView.tsx`의 `overlayContainer`에서
- 클릭 지점에 빨간 `AddIcon` 마커.
- 그 위에 거리 라벨(`predictedDistance`, `m` 단위) — `lime` 고정 색.
- `stop_scaled_uv` 위치에 빨간 `AddIcon` 마커(=base 정지 예상 위치).

### (B) Move Base 액션 — "Move Base!" 버튼  *(요구사항 1~4)*

- **액션**: `move_base_to_point` (`MoveBaseToPoint`)
  - 입력(goal): `scaled_u`, `scaled_v` — nav 카메라에서 클릭한 정규화 픽셀(0~1) *(1)*
  - 결과: `status` (SUCCESS / FAILURE / TIMEOUT / CANCELLED)
  - 피드백: `new_scaled_u/v`, `show_click_marker`, `new_stop_scaled_u/v`,
    `show_stop_marker`, `elapsed_time`.
- **처리 흐름**
  1. goal로 정규화 클릭 픽셀 수신. *(1)* 액션 수락 즉시 피드백을 1회 발행해
     **클릭 마커와 stop 마커를 모두 숨긴다** (`show_click_marker=false`,
     `show_stop_marker=false`).
  2. 팔 stow 동작. *(2)*
  3. base를 goal 방향으로 회전(`pan_tilt_offset_to_center`의 `delta_pan`)
     + `head_pan = 0.0`. *(3)*
  4. base translation = T0 base_link 의 클릭 3D 점에서 base 머리 도달 위치
     (= 클릭 방향으로 `BASE_MARGIN` 남기고 정지)를 계산해 `goal_pose` 를
     T0 base_link 로 구성. 이후 `tf2_transform` 으로 odom(world 고정) 으로
     변환해 `MOVE_BASE` state 에 넘긴다. `translate_base_to_goal_pose` 는 매
     iteration 현재 base_link 로 다시 변환해 err 를 계산하므로, base 가
     이동/회전해도 같은 world 점을 추종한다 (pregrasp 패턴). *(4)*
- **마커 표시 변화** *(요구사항 추가)*
  - 액션 수락 ~ rotation 완료: 두 마커 모두 숨김.
  - 마커가 가리키는 점 = "base_link 원점 + BASE_MARGIN 전방"의 고정 world
    점. 즉 base 머리(원점에서 BASE_MARGIN 앞)가 마커에 도달하면 액션 성공.
  - Rotation 완료 직후: stop 마커를 `post_click_uv` (nav 이미지 평면에서
    `DEPTH_REF` 주위로 클릭 픽셀을 image-forward 방향에 정렬하도록 회전한
    좌표)에 표시. 이 위치가 base 머리의 최종 목표 위치다.
  - Translation 진행 중: err_callback의 남은 거리(m)로
    `ratio = clamp((BASE_MARGIN + remaining) / raw_distance, 0, 1)`,
    `stop_uv = DEPTH_REF + ratio · (post_click_uv − DEPTH_REF)`.
    초기 ratio=1.0(post_click), 성공 시 ratio=BASE_MARGIN/raw 위치로 수렴
    (= base 머리가 도달한 지점, DEPTH_REF 보다 살짝 앞).
- 거리 계산 로직은 (A)/(B) 공통 (`compute_click_point_in_base`, raw 반환).
  팔 길이 차감과 후처리(마커 좌표 변환)는 호출자 측에서 수행.

### 구독 토픽
- `/navigation_camera/image_raw/rotated/compressed`

### 헬퍼 의존성
- `constants.py` (`NAV_CAMERA_K/D`, 신규 `TILT_READY`·`DEPTH_REF`)
- `functions.py` (`pan_tilt_offset_to_center` — base 회전 방위각)
- `da/` (Depth-Anything `get_pred_depth`)
- `move_to_action_state.py`, `stretch_ik_control.py`, `conversions.py`

### 기존 코드 대비 변경점
- `BASE_TRANSLATION`을 0.0 → pred_depth 기반 직선거리로 사용.
- 기존 head tilt 조준 로직(`nav_scaled_v_after_tilt`, VFOV focal length 계산)
  제거 → 고정 `TILT_READY` + `DEPTH_REF` 방식으로 단순화.
- `/get_distance` 서비스를 신설. **현재 이 서비스는 `move_gripper_to_point.py`가
  제공 중**이므로, `move_base_to_point.py`로 소유권을 옮기고 move_gripper 쪽
  중복 서비스는 제거/정리해야 한다 (두 노드가 동시에 같은 서비스명 등록 불가).

### 확인 필요 (open questions)
1. ~~거리 overlay 렌더링 위치~~ → **확정**: 웹 UI `CameraView.tsx`의
   `overlayContainer`(마커 + 거리 라벨). 노드는 `distance`(m)만 반환.
2. ~~`TILT_READY` 범위~~ → **확정**: head_tilt만, head_pan 미포함.
   (추후 head_pan별 가변 `DEPTH_REF`를 array로 확장 가능.)
3. ~~`DEPTH_REF` 표현 형태~~ → **확정**: `ref_rcz` 형태 `(row, col, depth_mm)`.
   상수명은 `DEPTH_REF`.
4. ~~Move Base 액션의 `head_tilt` 재확인~~ → **확정**: 재확인하지 않음
   (미리보기에서 이미 `TILT_READY`로 정렬됐다고 가정).
5. ~~base 회전 방위각 계산 방식~~ → **확정**: `functions.py`의
   `pan_tilt_offset_to_center`의 `delta_pan`을 base 회전각으로 사용.
6. ~~`move_to_action_state.py` 편집 범위~~ → **확정**: 개발 범위에 포함.
   `get_state_for_move_base_to_point()` 등 필요한 상태/헬퍼를 직접 수정한다.

### 정리 메모 / TODO
- `tkinter` import(`from tkinter import Y`) 미사용 — 제거.
- 기존 `x_dist`/`new_scaled_v` 관련 주석 코드 제거.
- 클래스/메서드 docstring 거의 없음.

## `move_gripper_to_point.py` — 그리퍼 이동 + 거리 측정

클릭한 지점으로 **그리퍼**를 이동시키고, 클릭 좌표까지의 거리를 추정해 제공한다.

- **액션**: `move_gripper_to_point` (`MoveGripperToPoint`)
  - 입력: `scaled_u`, `scaled_v` — 클릭한 정규화 픽셀
  - 처리: 클릭 좌표 → `BASE_ROTATION` + 헤드 `HEAD_PAN`(그리퍼 방향, -90°)/`HEAD_TILT`
    + `BASE_TRANSLATION`(raw 거리 − `DEFAULT_ARM_LENGTH` = 팔이 닿는 위치),
    `get_state_for_move_gripper_to_point()` 상태 머신으로 베이스·헤드·팔(lift/extension) 이동.
  - 클릭 3D 점은 `compute_click_point_in_base`(공용 helper)으로 base_link
    프레임에서 얻고, base 머리 도달 위치(= 클릭 방향으로 마진 남기고 정지)
    를 `goal_pose` 로 구성해 odom 으로 변환 (pregrasp 패턴, Move Base 와 동일).
    Move Base 와의 차이는 차감 마진뿐: Move Base는 `BASE_MARGIN`(0.45 m),
    Move Gripper는 `DEFAULT_ARM_LENGTH`(0.8 m).
- **서비스**: `get_distance` (`GetDistance`)
  - 입력: 클릭한 정규화 픽셀 → 출력: `distance`(m, 직선거리), `success`
  - 내비게이션 카메라 영상으로 Depth-Anything(`get_pred_depth`) 깊이 예측,
    RealSense depth 중앙 픽셀을 기준값으로 내비게이션 프레임에 정합
    (`align_realsense_ref_to_navigation`), 픽셀 ray 방향으로 직선거리 환산.
- **구독 토픽**
  - RealSense: `/camera/color/image_raw/compressed`,
    `/camera/aligned_depth_to_color/image_raw/compressedDepth`,
    `/camera/aligned_depth_to_color/camera_info`
  - 내비게이션: `/navigation_camera/image_raw/rotated/compressed`
  - 그리퍼: `/gripper_camera/image_raw/compressed`,
    `/gripper_camera/depth/image_rect_raw/compressedDepth`,
    `/gripper_camera/depth/camera_info`
- **부가 기능**: `save_images()` / `save_ggcnn_results()` — RGB·depth·예측depth 저장 및
  GG-CNN 그래스프 예측(quality/angle/width) 결과 저장(디버깅·데이터 수집용).
- **헬퍼 의존성**: `constants.py`(`NAV_CAMERA_K/D`, `NAV_OPTICAL_FROM_LINK`),
  `conversions.py`, `move_to_action_state.py`, `stretch_ik_control.py`,
  `ggcnn/`(그래스프 예측), `da/`(Depth-Anything 깊이 추정)

### 개발 메모 / TODO
- `tkinter` import 미사용 — 정리 필요.
- `get_optimal_grasp()`는 미구현(`return`만 존재) — GR-ConvNet 연동 예정.
- `get_clicked_pixel()`이 참조하는 `self.image_params`가 미정의 — 사용 시 초기화 필요.
- 코드 흐름상 미사용 메서드(`gripper_realsense_depth_cb`의 주석 처리된 대안 등) 정리 여지.
