# EX-CAD의 의미 기반 모델링

Blender 5.2.1 LTS에서 동작하는 로컬 브리지로, 스케치 치수와 피처 이름, 인물의 스타일, 물체 간 관계를 지정해 장면을 편집한다. 모델을 만든 창에서 피처 치수를 바꾸고 네 방향에서 결과를 확인할 수 있다.

## 기존 Blender에 연결하기

Windows 저장소 루트에서 `cmd /c install_blender_addon.bat`를 실행한다. Blender의 환경설정에서 `EX-CAD` 애드온을 활성화하면 3D 뷰포트의 EX-CAD 탭에서 브리지 상태를 볼 수 있다. 인물 기능에는 Blender Extensions의 MPFB가 필요하다. 기본 포트는 `127.0.0.1:9877`이며, 한 줄의 JSON 요청에 한 줄의 JSON으로 응답한다. 설치된 파일을 갱신한 뒤 같은 연결 주소로 `{"cmd":"reload_addon"}`을 보낸다.

MCP 클라이언트의 stdio 서버 명령에는 `python`을, 인수에는 저장소의 `excad_mcp_server.py` 절대 경로를 지정한다. `EXCAD_PORT`로 포트를 바꿀 수 있다.

공식 Python MCP SDK 1.30.0의 `initialize`와 2.3.0의 `server/discover`로 각각 네 도구를 호출해 PNG와 장면 복원을 확인했다. TypeScript SDK 1.32.2와 2.4.0 클라이언트에서도 같은 네 도구를 호출했다. Python으로 재현하려면 `EXCAD_TEST_MCP_PYTHONS`에 해당 SDK가 설치된 Python 실행 파일 경로를 설정한 뒤 아래 `layout.py` 테스트를 실행한다. Windows에서는 여러 경로를 `;`로 구분한다. [MCP 버전 협상 규격](https://modelcontextprotocol.io/docs/2026-07-28/learn/versioning)

| 도구 | 쓰임 |
| --- | --- |
| `excad_schema` | 빈 인수로 명령 목록을 읽는다. `topic`에 `sketch`, `cad`, `character`, `organic`, `scene` 중 하나를 주면 해당 필드를 볼 수 있다. |
| `excad_run` | `steps`에 명령 배열을 주거나 `code`에 DSL을 넣는다. 기본 응답은 실행 횟수와 마지막 결과다. 각 단계의 결과가 필요하면 `return_all`을 사용한다. |
| `excad_digest` | 처음 응답의 `h`를 다음 호출의 `since_hash`에 넣으면 바뀐 항목과 삭제된 이름을 받는다. 최근에 조회하거나 기준으로 사용한 해시 16개를 보관하며, 캐시에서 사라진 해시로 요청하면 전체 목록을 보낸다. |
| `excad_inspect` | `target`에 물체 이름이나 `ALL`을 넣는다. PNG 이미지, 저장 경로, 월드 좌표의 경계 상자와 진단 결과를 돌려준다. |

## 저장한 피처 레시피 수정하기

다음 예제는 저장소 루트에서 일반 Python으로 실행한다. 첫 배치가 장공을 통과시킨 플레이트를 만들고 재질과 프리뷰를 적용한다. 두 번째 요청은 본체 두께와 스케치 폭을 바꾸고 후속 피처를 재생성한다. [예제 레시피](feature_tree.json)

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, "excad_addon")
from client_compact import rpc

recipe = json.loads(Path("doc/excad/feature_tree.json").read_text(encoding="utf-8"))
created = rpc({
    "cmd": "batch",
    "steps": [
        {"cmd": "feature_tree", "name": "Housing", "recipe": recipe},
        {"cmd": "set_material", "object": "Housing", "preset": "BRUSHED_METAL"},
        {"cmd": "render_inspect", "target": "Housing"},
    ],
})
assert created["ok"], created
print(created["last"]["path"])

edited = rpc({
    "cmd": "rebuild_feature_tree",
    "object": "Housing",
    "parameters": {"pad.depth": 0.8, "sketches.plate.shapes.0.width": 3.4},
})
assert edited["ok"], edited
```

길이는 Blender 단위, 각도는 도 단위다. `SLOT.length`는 양쪽 반원을 포함한 전체 길이이고, `start`와 `end`를 주는 형식에서는 두 점이 반원의 중심이다. `SELF`는 레시피가 재생성하는 현재 본체다. `sketch_on_face`로 윗면을 선택하면 이후 피처는 그 면의 로컬 평면을 사용한다. 홈을 팔 때에는 음의 깊이를 준다. 무효한 치수로 피처가 실패하면 이전 솔리드가 유지된다. `save_feature_tree`는 JSON을 저장하고, `rebuild_feature_tree`의 `path`로 다시 불러올 수 있다.

`loft`는 단면의 `sketch_ids` 순서를 따르며 `SMOOTH`와 `LINEAR`를 지원한다. 여러 관통 구멍이 있는 단면은 구멍의 위치와 포함 관계로 연결되며 단면마다 직경과 그린 순서를 바꿀 수 있다. `sweep`는 한 개의 스플라인을 가진 Blender 커브 이름이나 경로 점 배열을 받는다. 이름으로 연결한 커브는 재생성 시 셰이프 키와 변형 모디파이어를 적용한 중심선을 따른다. 원본 커브가 삭제되면 마지막으로 만든 중심선을 사용한다. 저장한 JSON에는 그 점들이 포함되어 다른 장면에서도 같은 단면을 다시 뽑을 수 있다. `fillet`, `chamfer`, `shell`, `pattern`, `mirror3d`는 같은 트리에 순차적으로 추가된다. 스케치 편집 중 뷰포트에서 파란 선은 자유도가 남은 도형, 검은 선은 완전히 구속된 도형이다.

`FIXED`는 점을 현재 위치에 고정한다. 특정 위치에 고정하려면 `value`와 `value2`에 로컬 평면 좌표를 넣는다. 오프셋과 대칭 복사본은 원본의 구속 상태를 따르며, 장공에는 전체 길이와 폭이 표시된다.

## 사진에서 인물 만들기

```json
{
  "cmd": "batch",
  "steps": [
    {"cmd":"human_generate","name":"Portrait","rig":"game_engine"},
    {"cmd":"fit_face_from_image","object":"Portrait","image_path":"C:/reference/front.png"},
    {"cmd":"character_stylize","object":"Portrait","hair":"PONYTAIL","clothes":["TSHIRT","PANTS","SHOES"],"skin":"WARM_BEIGE","iris_color":[0.12,0.06,0.02],"iris_size":0.52,"lashes":true,"teeth":true,"expression":"SMILE","hands":"RELAXED"},
    {"cmd":"place_relative","obj":"Portrait","relation":"ALIGN_FLOOR"},
    {"cmd":"render_inspect","target":"Portrait"}
  ]
}
```

정면을 향한 얼굴이 한 명 나온 사진을 사용한다. 첫 실행은 로컬 Python 3.12로 MediaPipe 실행 환경을 준비하고 Face Landmarker 모델을 캐시한다. `EXCAD_FACE_PYTHON`과 `EXCAD_FACE_MODEL`에는 기존 인터프리터와 모델 경로를 지정할 수 있다. 사진에서 측정한 비율 8개를 양쪽에 대응하는 MPFB 모프에 적용한다. 다른 사진을 적용하면 눈의 중심과 크기도 바뀐다. 눈의 기준 형상을 파일에 보존하므로 두 사진을 번갈아 적용한 뒤 첫 사진으로 돌아오면 같은 눈 형상이 복원된다. [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

헤어, 상의, 겉옷, 하의와 신발은 슬롯별로 교체한다. `clothes: []`는 의상을 벗기고, `lashes: false`는 속눈썹을 제거한다. `iris_size`만 바꾸면 현재 홍채 색이 유지된다. 애드온 패널의 **Install CC0 character assets** 버튼과 `{"cmd":"install_asset_pack","pack":"all"}`은 시스템·셔츠·바지·치마 팩을 내려받아 해시를 확인한다. `equip_asset`에 `human`, `name`, 필요 시 `pack`을 주면 장착과 리핏을 실행한다. `EXCAD_CACHE`로 다운로드와 텍스처 미리보기의 캐시 위치를 지정할 수 있다. [MakeHuman 공식 에셋 팩](https://static.makehumancommunity.org/assets/assetpacks.html)

겉옷만 벗기려면 `{"cmd":"character_stylize","object":"Portrait","remove":["outerwear"]}`를 보낸다. 재킷 아래에 가려졌던 티셔츠나 스웨터가 다시 드러난다. 긴 머리는 머리 본을 따라 움직이며, 어깨를 올리면 Geometry Nodes가 현재 포즈의 옷 표면에 맞춰 머리끝을 조정한다.

상반신을 촬영하려면 `{"cmd":"light_studio","target":"Portrait","style":"HEADSHOT"}`을 보낸다. 현재 포즈의 양쪽 눈 위치와 간격을 기준으로 86 mm 카메라와 세 개의 소프트박스를 배치한다. 인물을 회전하거나 다른 포즈를 적용한 뒤 같은 명령을 보내면 기존 카메라와 조명을 다시 맞춘다. Blender에서 F12를 누르면 현재 렌더 엔진으로 촬영한다. 빠른 네 방향 검수에는 `render_inspect`를 사용한다.

## SDF, 다면 투영과 외부 3D 생성

```json
{
  "cmd":"sdf_generate","name":"Planter","resolution":96,"remesh":"QUAD",
  "tree":{"op":"smooth_subtract","k":0.035,"children":[
    {"type":"cone","height":0.88,"radius":0.38,"top_radius":0.57},
    {"type":"cone","height":0.93,"radius":0.32,"top_radius":0.52,"center":[0,0,0.10]}
  ]}
}
```

`resolution`은 가장 긴 축의 격자 해상도다. `QUAD`는 표면을 사각면으로 추출하고 `VOXEL`은 Blender의 복셀 리메시를 적용한다. 사각면에 비매니폴드 연결이 생긴 경우에도 복셀 리메시를 거쳐 솔리드를 검사한다. 각 노드는 `scale`, XYZ 기준 `rotation_deg`, `center` 순서로 변환된다. 예를 들어 토러스에 `rotation_deg: [90,0,0]`과 `scale: [1.2,1,1]`을 주면 손잡이에 쓸 수 있는 세로 고리가 된다.

저장한 `.blend`를 다시 연 뒤에도 아래 명령으로 화분의 안쪽과 바깥쪽 윗반지름을 함께 바꿀 수 있다. 새 메시로 UV, 면별 재질, 정점 색과 웨이트를 옮긴다. 같은 메시를 공유하던 다른 인스턴스는 원래 치수를 유지한다. SDF 위에 스케치로 구멍이나 돌출을 추가한 모델에서는 SDF 치수를 바꿀 때 CAD 피처도 순서대로 다시 실행한다. `save_feature_tree`로 저장한 JSON에 두 레시피가 함께 들어간다. 셰이프 키가 있는 메시의 SDF 갱신은 정점 대응을 보존할 수 없어 장면 변경 전에 오류를 반환한다.

```json
{"cmd":"sdf_generate","object":"Planter","parameters":{"children.0.top_radius":0.64,"children.1.top_radius":0.59}}
```

다면 투영의 `poly_reconstruct`에는 똑같은 중심과 크기에 정렬된 `face_00.png` 등의 이미지를 준다. 선택적인 `face_00_depth.png`는 같은 해상도의 선형 깊이로, 정사영 박스의 가까운 면이 0이고 먼 면이 1이다. `poly_bake`는 Smart UV와 법선 가중치로 기본 2048 × 2048 PNG를 만든다. 선형 색 공간에서 여러 시점을 혼합한 결과를 sRGB 이미지로 저장한다. `use_depth: false`로 색 이미지만 쓸 수 있다. 공유 메시의 한 인스턴스를 베이크하면 그 인스턴스에 새 UV와 재질이 적용된다.

기존 장면에서 입력을 만들 때는 아래 요청으로 정렬된 RGBA 이미지 20장과 16비트 깊이 이미지 20장을 함께 저장한다. `center`에는 촬영 범위의 중심을, `size`에는 한 변의 길이를 준다. 재구성에도 같은 값을 사용한다. `poly_render`는 현재 렌더 엔진으로 별도 장면에서 촬영한 뒤 기존 카메라·컴포지터·출력 설정으로 돌아온다. [Blender 렌더 패스](https://docs.blender.org/manual/en/5.2/render/layers/passes.html)

```json
{"cmd":"poly_render","output_dir":"C:/reference/vessel","poly_type":"ICOSAHEDRON_20","center":[0,0,0.6],"size":1.8,"img_res":256,"write_depth":true}
```

| `generate_3d.provider` | 입력과 연결 |
| --- | --- |
| `MESHY` | 텍스트 또는 이미지. Blender 프로세스에 `MESHY_API_KEY`를 지정한다. [Meshy API](https://docs.meshy.ai/en/api/text-to-3d) |
| `TRIPO` | 텍스트 또는 이미지. `TRIPO_API_KEY`를 사용한다. [Tripo API](https://developers.tripo3d.ai/en/docs/generation-image-to-model/standard) |
| `HUNYUAN3D-2` | `HUNYUAN3D_URL`의 기본값은 `http://127.0.0.1:8080`이다. 텍스트 입력에는 서버의 텍스트 변환 파이프라인이 필요하다. 텍스처까지 요청하려면 모델 서버를 `--enable_tex`로 시작하고 `texture: true`를 보낸다. [Hunyuan3D-2 서버](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/api_server.py) |
| `TRELLIS` | Gradio 이미지 서버에 `image_path`를 전달한다. 원본 서버의 두 단계 생성 방식과 커뮤니티 서버의 `generate_and_extract_glb` 방식을 지원한다. `TRELLIS_URL`의 기본값은 `http://127.0.0.1:7860`이다. [TRELLIS 서버](https://github.com/microsoft/TRELLIS/blob/main/app.py) |

Hunyuan3D-2의 모델 라이선스는 한국·EU·영국을 적용 지역에서 제외한다. [Tencent 모델 라이선스](https://huggingface.co/tencent/Hunyuan3D-2mini/blob/main/LICENSE)

생성된 GLB는 지정한 `height`로 스케일을 맞추고 바닥에 접지한다. 여러 부품이 있으면 반환값의 `name`으로 조립체 전체를 이동할 수 있다. `rig: "PARTS"` 또는 `rig: true`는 리그가 없는 부품에 본과 웨이트를 부여한다. 사람 형태에는 `rig: "HUMANOID"`를 지정한다. 머리·몸·의복이 별도 메시인 인물에도 하나의 스켈레톤을 만들고 같은 웨이트를 의상과 장식에 전이한다. 원본 GLB에 리그가 있으면 그 리그를 사용하며 루트 애니메이션의 이동과 크기를 유지한다. `remesh: true`는 복셀 리메시를 적용한다. `timeout`의 단위는 초이며 기본값은 900이다. 원격 호출과 다운로드는 작업 스레드에서 처리하고 Blender 메인 스레드에서 장면에 가져온다.

복셀 리메시를 선택하면 새 면마다 원본의 UV 섬을 찾고, 그 안에서 각 모서리의 텍스처 좌표를 보간한다. 서로 떨어진 무늬 사이를 가로지르는 픽셀을 줄이고 여러 UV 레이어와 활성 렌더 레이어를 보존한다.

## 배치 결과 확인과 복구

`place_relative`는 이름으로 지지물과 이동 대상을 지정한다. `ON_TOP_OF`는 양쪽의 실제 표면을 기준으로 수직 간격을 맞춘다. `scatter_on_surface`는 면적 비례 샘플링을 사용하며 `seed`로 같은 배치를 재현할 수 있다. `resolve_overlaps`의 `fixed`에 움직이면 안 되는 물체를 넣는다. 부모 아래의 부품은 기본적으로 한 조립체로 이동한다.

굴곡이 있는 지형에 물체를 흩뿌릴 때 `scatter_on_surface`는 각 사본의 밑면과 지형이 실제로 닿는 위치를 구한다. `max_slope: 40`을 지정하면 월드 위쪽을 기준으로 경사 40도 이하인 면만 사용한다. 닫힌 지형의 밑면과 옆면을 제외하고 소품을 세울 때 쓸 수 있다. `align_normal: false`를 함께 주면 각 사본이 원래 자세를 유지한다.

`render_inspect`는 평가된 메시로 정면·측면·사선·상단을 그린다. `warnings`에 겹침, 부유, 비정상적으로 얇은 형상, 열린 솔리드가 표시된다. 인물의 장착 부위와 옷 끝단의 접촉은 `contacts`, 헤어 카드와 의복의 열린 경계 수는 `surface_topology`에서 확인할 수 있다. 기본 타일은 160 × 160이며 모델 복잡도와 타일 크기에 따라 처리 시간이 달라진다.

받침대 위에 놓인 부품 하나를 지정하면 이미지에는 지정한 부품이 나오고, 부유 검사는 실제 받침대까지 확인한다. `set_material`로 바꾼 광택과 금속성, `light_studio`의 세 조명도 프리뷰에 반영된다. 기본 `scene_digest`에서 생성한 조명과 정사영 카메라의 변경을 추적할 수 있다.

같은 메시를 여러 번 배치하면 프리뷰가 각 인스턴스의 변환·재질을 적용하고 삼각형 검색 구조를 공유한다. Windows의 Blender 5.2.1에서 받침대와 색상이 다른 타원체 196개(삼각형 약 78만 개)를 검사한 두 번의 측정은 377 ms와 384 ms였다. 두 번 모두 겹친 한 쌍과 떠 있는 한 물체를 찾았다.

생성한 인물과 불러온 GLB에는 텍스처의 검사용 축소본도 저장한다. 파일을 다시 열면 원본 이미지 내용, 색 공간, 알파 처리 방식이 일치하는 축소본을 사용한다. 아래 도자기 전시 장면을 네 번 각각 빈 캐시에서 열었을 때 첫 네 방향 프리뷰는 417~554 ms가 걸렸다. 기본 타일 크기는 160픽셀이다.

`checkpoint`에 이름을 주고 편집한 뒤 `rollback`에 같은 이름을 보내면 활성 장면이 복원된다. `undo`는 최근 브리지 편집을 한 요청 단위로 되돌린다. 여러 단계가 있는 배치는 처음 실패한 단계에서 중지되며 그 전에 성공한 편집은 유지된다. 전체 배치를 되돌리려면 `undo`를 사용한다.

## Blender 백그라운드 검증

저장소 루트에서 애드온을 동기화한 후 동일한 버전의 Blender로 아래 테스트를 실행한다. 사진 테스트는 `EXCAD_TEST_PHOTO`에 정면 사진 경로와 활성 MPFB 확장이 필요하다. `EXCAD_TEST_OUTPUT`을 지정하면 네 단계의 PNG를 같은 폴더에 저장한다.

```powershell
cmd /c install_blender_addon.bat
$blenderExe = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$env:EXCAD_TEST_VARIANTS = Join-Path $env:TEMP 'excad-character-variants'
foreach ($suite in 'protocol', 'cad', 'character', 'organic', 'multiview', 'view_capture', 'import_rig', 'layout', 'dense_layout', 'linked_inspection', 'scatter_terrain') {
    & $blenderExe -b --factory-startup --python-exit-code 1 --python "tests/excad/$suite.py"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $blenderExe -b --factory-startup --python-exit-code 1 --python tests/excad/endurance.py -- 71127038
& $blenderExe -b --factory-startup --python-exit-code 1 --python tests/excad/overlay.py
& $blenderExe -b --factory-startup --python-exit-code 1 --python tests/excad/surface_transfer.py
& $blenderExe -b --factory-startup --python-exit-code 1 --python tests/excad/sdf_rebuild.py
& $blenderExe -b (Join-Path $env:EXCAD_TEST_VARIANTS 'dressed.blend') --python-exit-code 1 --python tests/excad/portrait_render.py
& $blenderExe -b --factory-startup --python-exit-code 1 --python tests/excad/wardrobe_motion.py
& $blenderExe -b --factory-startup --python-exit-code 1 --python tests/excad/wardrobe_motion.py -- 81128002 81128003 81128007 81128018 81128021 81128053
```

`overlay.py`는 GPU가 있는 머신에서 창을 열지 않고 실제 GPU 프레임 버퍼에 선과 치수를 그린다. `tests/excad/trellis_transport.py`는 TRELLIS의 Gradio 4.44.1 및 5.34.2 환경에서 일반 Python으로 실행한다. 이 테스트는 한 세션 안에서 이미지 업로드·전처리·3D 작업·GLB 추출까지 이어지는 전송 경로를 검사한다. 모델 추론에는 별도로 실행 중인 모델 서버나 서비스 계정이 필요하다.

## 모델과 프리뷰

| 모델 | 확인할 수 있는 부분 |
| --- | --- |
| [파라메트릭 플레이트](previews/phase1-cad.png) · [스케치 오버레이](previews/phase1-sketch-overlay.png) | 장공과 모서리 가공, 고정 도형과 자유 도형의 선색, 치수 |
| [사진에 맞춘 인물](previews/phase2-character.png) · [Cycles 상반신 렌더](previews/phase2-portrait.png) | MPFB 얼굴 모프, 홍채와 각막, 피부·헤어 셰이더, 셔츠·바지·신발의 피팅 |
| [SDF 꽃병](previews/phase3-sdf.png) · [20면 UV 복원](previews/phase3-uv.png) | 용기의 열린 입구와 내부 형상, 비대칭 유약 무늬의 정렬 |
| [TRELLIS 입력](previews/trellis-reference.png) · [실제 생성 결과](previews/trellis-generated.png) | Cycles로 렌더한 꽃병에서 생성한 GLB. 색상은 반영됐으며 좁은 목과 굴곡은 원통에 가깝게 단순화됐다. |
| [받침대 위 화분](previews/phase4-layout.png) | 서로 다른 두 물체의 접지, 목재와 도자기 재질, 스튜디오 조명 |
| [도자기 전시 장면의 Cycles 렌더](previews/ceramics-exhibit.png) | 사진 피팅 인물, CAD 받침대 두 개, SDF 그릇, TRELLIS 꽃병을 한 장면에 조립한 결과 |

전시 장면에서 왼쪽 받침대의 `height.depth`를 0.72에서 0.81로 바꾸고 SDF 그릇의 입구를 넓힌 뒤 `place_relative`로 다시 접지했다. 저장한 파일을 다시 연 상태에서 공식 MCP 클라이언트로 편집했고, 그릇 바닥의 높이 0.81과 롤백 후 원래 장면 해시를 확인했다.

이 장면의 19개 물체를 요약한 전체 JSON 응답은 1,727바이트다. 받침대 높이와 그릇 위치를 한 배치로 바꾸면 `since_hash` 응답은 두 물체의 변경분 237바이트를 담는다. 편집을 되돌린 뒤 원래 해시로 조회한 응답은 47바이트다. 크기는 브리지 응답의 UTF-8 기준이다.

TRELLIS 실측은 공개 Gradio 모델 서버에 합성 꽃병 이미지 한 장을 보낸 결과다. 생성과 다운로드에는 75.8초가 걸렸다. Blender에서 텍스처 경계의 정점을 연결한 결과 정점 5,885개·삼각형 11,766개이며 열린 경계와 비매니폴드 모서리는 0개였다. [검증에 사용한 TRELLIS 서버](https://huggingface.co/spaces/trellis-community/TRELLIS)

인물 예제의 얼굴 비율은 scikit-image가 제공하는 NASA의 에일린 콜린스 사진에서 측정했다. 머리 모양, 의상, 피부색은 별도 스타일 파라미터로 정했다. [검증용 사진 출처](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)
