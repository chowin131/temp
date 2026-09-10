# CNN architecture 비교 실험

CIFAR-10 / CIFAR-100에서 ResNet, PreActResNet, DenseNet, FractalNet, ViT를
같은 조건으로 학습시켜 비교하는 코드.

## 폴더 구조

```
main.ipynb        코랩 진입점. 셀에서 main.run_experiment(...) 호출
main.py           재료(dataloader/encoder/method/optimizer)를 만들어 trainer.fit에 넘김
trainer.py        학습/평가 루프
datasets.py       CIFAR-10 / CIFAR-100 dataloader
utils.py          결과 불러오기, 그래프
architectures/    classifier 없는 encoder들
methods/          encoder를 감싸서 loss를 만드는 학습 방식
runs/             결과 json + 베스트 체크포인트 (자동 생성)
data/             CIFAR 원본 (자동 다운로드)
```

핵심 규칙 두 가지.

1. `architectures/`의 클래스는 **classifier가 없는 encoder**다.
   `forward(x)`가 feature 벡터를 돌려주고 그 차원을 `num_features`로 알려준다.
2. `methods/`의 클래스는 encoder를 감싸는 `nn.Module`이고
   **`forward(batch)`가 loss를 돌려준다.** classifier도 여기 붙는다.
   그래서 optimizer는 `method.parameters()`를 그대로 받으면 되고,
   학습 루프는 `loss = method(batch)` 한 줄만 알면 된다.

## 시작하기

1. 폴더 전체를 코랩에 업로드하고 `main.ipynb`를 연다.
2. 런타임 → 런타임 유형 변경 → **GPU (T4)**.
3. 위에서부터 셀을 실행. 0~1번 섹션에서 GPU가 잡혔는지, 뭐가 있는지 확인.
4. 2번 섹션은 `run_experiment`가 안에서 하는 일을 펼쳐놓은 셀이다.
   학습 한 스텝이 `loss = method(batch)` → `loss.backward()` 뿐이라는 걸
   확인할 수 있고, 여기까지 돌면 나머지 파이프라인도 다 도는 것이다.
5. 그 다음 3번 섹션에서 원하는 실험 셀만 실행.

**결과를 Drive에 저장하는 걸 권장한다.** 코랩 런타임이 끊기면 `runs/`가
통째로 날아간다.

```python
from google.colab import drive
drive.mount('/content/drive')

RUNS = "/content/drive/MyDrive/ell_runs"
result, best_acc = main.run_experiment(arch="resnet", out_dir=RUNS)
```

## 실험 돌리기

```python
import main

result, best_acc = main.run_experiment(
    arch="resnet", blocks=[3, 3, 3], run_name="resnet20",
)
```

`run_experiment`는 `(result, best_acc)`를 돌려준다. **노트북에서는 반드시
변수로 받자.** 그냥 호출하면 Jupyter가 164개짜리 리스트 4개를 통째로 출력한다.

### 인자

전부 기본값이 있으니 바꿀 것만 넘기면 된다.

| 인자 | 기본값 | 설명 |
|---|---|---|
| `arch` | `"resnet"` | encoder 종류. `architectures.ENCODER_REGISTRY`의 키 |
| `blocks` | `None` | 깊이. **arch마다 의미가 다름 (아래 참고)**. `None`이면 `architectures.DEFAULT_BLOCKS`의 값 |
| `arch_kwargs` | `None` | 아키텍처 추가 인자. 예: `{"option": "C"}`, `{"k": 12}` |
| `method_name` | `"supervised"` | 학습 방식. `methods.METHOD_REGISTRY`의 키 |
| `dataset` | `"cifar10"` | `cifar10` / `cifar100` |
| `data_root` | `"data"` | CIFAR 원본 위치 |
| `batch_size` | `128` | |
| `num_workers` | `2` | 코랩에서는 2가 무난 |
| `epochs` | `164` | |
| `lr` | `0.1` | SGD 초기 learning rate |
| `momentum` | `0.9` | |
| `weight_decay` | `1e-4` | |
| `milestones` | `(82, 123)` | 이 epoch에서 lr에 `gamma`를 곱함 |
| `gamma` | `0.1` | |
| `seed` | `42` | 매 실험 시작 때 다시 심음 |
| `amp` | `True` | mixed precision. cuda일 때만 실제로 켜짐 |
| `out_dir` | `"runs"` | 결과/체크포인트 저장 폴더 |
| `run_name` | `None` | 결과 파일 이름. `None`이면 `arch_blocks_dataset` |
| `log_interval` | `100` | 배치 로그 간격. `0`이면 끔 |

### `blocks`의 의미가 arch마다 다름

가장 헷갈리는 부분이다. 리스트의 **길이**는 언제나 block(stage) 개수고,
**각 원소**의 뜻이 다르다.

| arch | 원소의 뜻 | 예시 |
|---|---|---|
| `resnet`, `preactresnet` | block당 layer 수 | `[3,3,3]` → 20 layer |
| `densenet` | block당 dense layer 수 | `[16,16,16]` → DenseNet-BC-100 |
| `fractalnet`, `fractalnet_loop` | block당 column 수 C | `[3]*5` → 20 layer |
| `vit` | encoder block 수 (원소 1개만) | `[12]` → ViT-Ti/4 |

FractalNet은 block마다 max-pool을 한 번씩 하므로 32×32 입력에는
block이 최대 5개다. `[3,3,3,3,3]`이 32→16→8→4→2→1로 딱 맞는다.

### 권장 순서

1. **ResNet 옵션 비교** (A/B/C) — 셋 다 ResNet-20이라 제일 빠름
2. **깊이 비교** — ResNet-56, ResNet-110
3. **PreActResNet** — ResNet과 같은 깊이로 짝 맞춰서
4. **DenseNet, FractalNet, ViT** — 오래 걸리니 마지막에
5. **결과 비교** (4번 섹션) — `runs/`의 json을 다 읽어서 표·그래프

처음 한 번은 `epochs=2, milestones=[1]`로 짧게 돌려보면 파이프라인이 도는지
1~2분 만에 확인할 수 있다.

### 모델 크기와 대략적인 소요 시간

파라미터 수는 실측값(CIFAR-10, classifier 포함)이고, **시간은 T4 기준
어림짐작**이다. 정확한 값은 첫 epoch가 끝나면 찍히는 `ETA` 줄을 보면 된다.

| 모델 | `arch` | `blocks` | 파라미터 | 164 epoch 예상 |
|---|---|---|---|---|
| ResNet-20 | `resnet` | `[3, 3, 3]` | 272,474 | ~1시간 |
| ResNet-56 | `resnet` | `[9, 9, 9]` | 855,770 | ~2시간 |
| ResNet-110 | `resnet` | `[18, 18, 18]` | 1,730,714 | ~3시간 |
| PreActResNet-20 | `preactresnet` | `[3, 3, 3]` | 272,282 | ~1시간 |
| PreActResNet-110 | `preactresnet` | `[18, 18, 18]` | 1,730,522 | ~3시간 |
| DenseNet-BC-100 | `densenet` | `[16, 16, 16]` | 769,162 | ~4시간 (300 epoch 권장) |
| FractalNet C=3, B=5 | `fractalnet` | `[3, 3, 3, 3, 3]` | 4,328,506 | ~2시간 |
| FractalNet (반복문) | `fractalnet_loop` | `[3, 3, 3, 3, 3]` | 4,328,506 | ~2시간 |
| ViT-Ti/4 | `vit` | `[12]` | 5,362,762 | ~1.5시간 |

두 FractalNet 구현은 파라미터 수가 정확히 같고, drop-path를 끈 eval 모드에서
같은 입력에 같은 출력을 낸다 (재귀 버전의 가중치를 반복문 버전에 옮겨 심어서
확인했고, 출력 차이는 0이었다).

## 결과 보기

실험 하나가 끝나면 `out_dir`에 두 파일이 생긴다.

- `{run_name}.json` — epoch별 `train_loss`, `test_loss`, `test_acc`, `lr`.
  **매 epoch 저장**되므로 중간에 끊겨도 거기까지의 곡선은 남는다.
- `{run_name}.pth` — test accuracy가 가장 좋았던 시점의 `state_dict`.
  encoder와 classifier가 같이 들어있다(method 전체).

노트북 4번 섹션이 `runs/*.json`을 전부 읽어서 요약표와 그래프를 그린다.

체크포인트를 다시 불러올 때는 학습 때와 **같은 arch/blocks/arch_kwargs**로
만들어야 한다.

```python
import torch
import architectures
import methods

encoder = architectures.get_encoder("resnet", [3, 3, 3])
method = methods.get_method("supervised", encoder, num_classes=10)
method.load_state_dict(torch.load("runs/resnet20.pth"))
method.eval()
```

## 주의사항

- **잘못된 이름을 넘기면 `KeyError`가 난다.** 오타 검사를 따로 하지 않으므로
  `arch="resnet20"` 같은 걸 넘기면 `KeyError: 'resnet20'`이 뜬다.
  `architectures.ENCODER_REGISTRY`의 키를 보고 맞춰 쓰면 된다.
- **ViT는 lr을 낮춰야 한다.** optimizer가 SGD로 고정돼 있는데, CIFAR 크기
  데이터를 처음부터 학습할 때 lr 0.1이면 잘 안 붙는다. `lr=0.01,
  weight_decay=5e-5` 정도로 돌린다. AdamW + warmup + cosine이 필요하면
  `main.py`의 optimizer/scheduler 두 줄을 고치면 된다.
- **`get_encoder`는 `blocks`를 반드시 받는다.** 기본값을 쓰고 싶으면
  `architectures.DEFAULT_BLOCKS[arch]`를 직접 넘긴다.
  `run_experiment`는 이걸 대신 해준다.
- **학습 재개는 안 된다.** 체크포인트에 optimizer 상태와 epoch이 없어서,
  세션이 끊기면 처음부터 다시 돌려야 한다. 오래 걸리는 실험은 `out_dir`을
  Drive로 잡아두는 편이 안전하다.
- **DenseNet은 논문 설정이 300 epoch**(`milestones=[150, 225]`)이다.
  164 epoch로 돌리면 다른 모델과 조건은 같아지지만 논문 수치와는 달라진다.
- **train dataloader는 `drop_last=True`다.** 마지막 자투리 배치를 버리므로
  실제 학습 샘플 수가 50,000보다 조금 적다.
- **재현성은 완벽하지 않다.** `seed`를 심고 `cudnn.benchmark=True`를 쓰기
  때문에, 같은 seed라도 GPU가 다르면 결과가 조금씩 달라질 수 있다.

## 코드 추가하기

### encoder 추가

1. `architectures/MyNet.py`에 클래스를 만든다. `__init__`의 첫 인자는
   `blocks`, `forward(x)`는 feature 벡터를 돌려주고, `self.num_features`를
   설정한다. classifier는 넣지 않는다.
2. `architectures/__init__.py`의 `ENCODER_REGISTRY`와 `DEFAULT_BLOCKS`에
   한 줄씩 추가한다.

### method 추가

1. `methods/my_method.py`에 `nn.Module`을 만든다.
   `__init__(self, encoder, num_classes)`, `forward(batch) -> loss`,
   그리고 평가용 `evaluate(batch) -> (loss 합, 맞힌 개수, 샘플 수)`.
2. `methods/__init__.py`의 `METHOD_REGISTRY`에 추가한다.

둘 다 `trainer.py`는 건드릴 필요 없다.
