# 📄 ResNet 논문 분석 (Deep Residual Learning for Image Recognition)

> He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.

---

## 목차

1. [Abstract (3줄 요약)](#1-abstract-3줄-요약)
2. [Introduction + Conclusion 요약](#2-introduction--conclusion-요약)
3. [핵심 구조 설명](#3-핵심-구조-설명)
4. [논문 ↔ 코드 연결 (PyTorch)](#5-논문--코드-연결-pytorch)
5. [데이터 흐름 추적 (Shape 변화)](#6-데이터-흐름-추적-shape-변화)
6. [면접 예상 질문 5개](#7-면접-예상-질문-5개)

---

## 1. Abstract (3줄 요약)

| | 내용 |
|---|---|
| **문제** | 신경망을 깊게 쌓을수록 오히려 학습이 안 되는 **성능 저하(degradation)** 현상이 발생한다. |
| **방법** | 층(layer)이 정답을 직접 찾는 대신, "원래 입력값과의 차이(잔차, residual)"만 학습하도록 구조를 바꾸고, 입력값을 출력에 그대로 더해주는 **숏컷 연결(shortcut connection)**을 추가했다. |
| **결과** | 152층짜리 초심층 네트워크를 성공적으로 학습시켜 ImageNet 대회에서 오류율 **3.57%**로 1위를 달성했다. |

---

## 2. Introduction + Conclusion 요약

### 왜 이 논문이 나왔나?

딥러닝에서 "층(layer)이 깊을수록 좋다"는 건 상식이었습니다. 층이 많을수록 더 복잡한 패턴을 배울 수 있거든요. 그런데 문제가 생겼습니다.

### 기존 방식의 문제점

**기울기 소실(vanishing gradient) 문제**라는 게 있었습니다. 쉽게 말하면, 학습 신호가 뒤쪽 층에서 앞쪽 층으로 전달되는데, 층이 너무 많으면 신호가 전달되는 도중 점점 약해져서 앞쪽 층은 아무것도 배우지 못하는 현상입니다.

이 문제는 **배치 정규화(Batch Normalization)** 같은 기법으로 어느 정도 해결됐습니다. 그런데 그 이후에 또 다른 문제가 발견되었습니다.

바로 **성능 저하(degradation)** 문제입니다. 층을 더 쌓으면 오히려 훈련 오류(training error)가 높아지는 현상인데, 이건 과적합(overfitting) 때문이 아닙니다. 진짜로 학습 자체가 잘 안 되는 것이었습니다.

> 논문의 Figure 1을 보면, **56층 네트워크**가 **20층 네트워크**보다 오히려 성능이 나쁩니다. 이론적으로 더 깊은 네트워크는 얕은 네트워크가 하는 일을 그대로 따라할 수 있어야 하는데, 실제로는 그렇게 되지 않았습니다.

### 이 논문의 해결책

**핵심 아이디어:** 층이 "정답"을 직접 학습하는 게 아니라, "원래 입력과 정답의 차이(잔차, residual)"만 학습하게 만들자!

```
기존 방식:  y = H(x)        → 정답을 직접 학습
ResNet 방식: y = F(x) + x   → 수정분(잔차)만 학습, x는 숏컷으로 그대로 전달
```

이 `x`를 출력에 더해주는 연결을 **숏컷 연결(shortcut connection)** 또는 **스킵 연결(skip connection)** 이라고 부릅니다.

---

## 3. 핵심 구조 설명

### ① 합성곱 층 (Convolutional Layer)

> **비유:** 사진에서 특정 패턴(가장자리, 질감, 색상)을 찾아내는 "돋보기". 이미지 위를 훑으면서 패턴이 있는지 체크합니다.

| 항목 | 내용 |
|---|---|
| **입력** | `(batch, 채널, 높이, 너비)` — 예) `(32, 3, 224, 224)` |
| **출력** | `(batch, 필터 수, 새 높이, 새 너비)` |
| **하이퍼파라미터** | 필터 크기, 필터 개수, 스트라이드(stride), 패딩(padding) |

---

### ② 배치 정규화 (Batch Normalization)

> **비유:** 시험 점수를 과목마다 "평균 0, 표준편차 1"로 표준화하는 것처럼, 각 층의 출력값 분포를 일정하게 맞춰줍니다.

| 항목 | 내용 |
|---|---|
| **입력** | 합성곱 층의 출력 텐서 |
| **출력** | 정규화된 텐서 (shape 동일) |
| **하이퍼파라미터** | 없음 (γ, β는 학습되는 파라미터) |

---

### ③ ReLU 활성화 함수

> **비유:** "0 이하는 무시하고, 0 이상은 그대로 통과"시키는 필터. 음수는 전부 0으로 만듭니다.

| 항목 | 내용 |
|---|---|
| **입력/출력** | 같은 shape, 음수 값만 0으로 변환 |
| **하이퍼파라미터** | 없음 |

---

### ④ 잔차 블록 (Residual Block) 🔴 핵심

> **비유:** 학생이 시험 답안을 처음부터 쓰는 게 아니라, "기존 답안에서 틀린 부분만 수정"하는 방식. 완전히 새 답안을 쓰는 것보다 훨씬 쉽겠죠?

**두 가지 버전:**

#### Basic Block (ResNet-18/34용)
```
Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+x) → ReLU
```

#### Bottleneck Block (ResNet-50/101/152용) 🔴
```
Conv(1×1) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1) → BN → (+x) → ReLU
```
- 1×1 conv로 채널을 먼저 **줄이고(병목)**, 3×3 conv로 처리 후, 다시 1×1로 **채널을 늘림**
- 계산량을 줄이면서도 깊은 네트워크를 만들 수 있음

| 항목 | 내용 |
|---|---|
| **입력/출력** | `(batch, C, H, W)` → shape 동일 (채널 변화 시 1×1 conv로 조정) |
| **하이퍼파라미터** | 블록 반복 횟수, 필터 수 |

---

### ⑤ 전역 평균 풀링 (Global Average Pooling)

> **비유:** 반 전체 학생 점수를 평균 하나로 요약하듯, H×W 크기의 특징 맵을 픽셀 하나로 압축합니다.

| 항목 | 내용 |
|---|---|
| **입력** | `(batch, C, H, W)` |
| **출력** | `(batch, C, 1, 1)` → flatten → `(batch, C)` |

---

### ⑥ 완전연결층 + Softmax

> **비유:** 여러 단서를 종합해서 "고양이 80%, 강아지 15%, 새 5%"처럼 최종 분류를 내리는 "판사" 역할.

| 항목 | 내용 |
|---|---|
| **입력** | `(batch, 2048)` |
| **출력** | `(batch, 1000)` — 1000개 클래스 각각의 확률값 |

---

## 5. 논문 ↔ 코드 연결 (PyTorch)

### 논문 → 코드 매핑

| 논문 표현 | PyTorch 코드 |
|---|---|
| `7×7 conv, 64 filters, stride 2` | `nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)` |
| `BN right after each convolution` | `nn.BatchNorm2d(64)` |
| `ReLU activation` | `nn.ReLU(inplace=True)` |
| `3×3 max pool, stride 2` | `nn.MaxPool2d(kernel_size=3, stride=2, padding=1)` |
| `F(x) + x (shortcut connection)` | `out += self.shortcut(x)` |
| `1×1 conv to match dimensions` | `nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)` |
| `global average pooling` | `nn.AdaptiveAvgPool2d((1, 1))` |
| `1000-way FC with softmax` | `nn.Linear(2048, 1000)` |

---

### 전체 ResNet-50 구현 코드

```python
import torch
import torch.nn as nn

# ① 병목 블록 (Bottleneck Block) - ResNet-50/101/152용
class BottleneckBlock(nn.Module):
    expansion = 4  # 출력 채널 = 입력 채널 × 4

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()

        # 논문의 F(x) 부분 (잔차 학습)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # 숏컷 연결 - 채널 수나 크기가 다를 때 1×1 conv로 맞춤
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != mid_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * self.expansion)
            )

    def forward(self, x):
        identity = x  # 원래 입력값 저장 (숏컷용)

        out = self.relu(self.bn1(self.conv1(x)))   # 1×1 conv
        out = self.relu(self.bn2(self.conv2(out))) # 3×3 conv
        out = self.bn3(self.conv3(out))            # 1×1 conv (ReLU 전)

        # 핵심! F(x) + x
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


# ② 전체 ResNet-50 모델
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # conv1
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x ~ conv5_x
        self.layer1 = self._make_layer(64,   64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(256,  128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512,  256, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, mid_channels, blocks, stride):
        layers = [BottleneckBlock(in_channels, mid_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(mid_channels * 4, mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # conv1
        x = self.layer1(x)  # conv2_x
        x = self.layer2(x)  # conv3_x
        x = self.layer3(x)  # conv4_x
        x = self.layer4(x)  # conv5_x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 사용 예시
model  = ResNet50(num_classes=1000)
dummy  = torch.randn(1, 3, 224, 224)  # 이미지 1장
output = model(dummy)
print(output.shape)  # → torch.Size([1, 1000])
```

---

## 6. 데이터 흐름 추적 (Shape 변화)

ResNet-50 기준, 이미지 1장(`batch=1`)이 통과할 때의 shape 변화입니다.

```
(1, 3, 224, 224)
    ↓ [conv1: 7×7, stride 2, 64 filters]
(1, 64, 112, 112)     # 224÷2 = 112, 채널: 3 → 64
    ↓ [MaxPool: 3×3, stride 2]
(1, 64, 56, 56)       # 112÷2 = 56
    ↓ [conv2_x: Bottleneck ×3, 채널 64→256]
(1, 256, 56, 56)      # 크기 유지, 채널 64×4=256으로 확장
    ↓ [conv3_x: Bottleneck ×4, stride 2, 채널 128→512]
(1, 512, 28, 28)      # 56÷2=28, 채널 128×4=512
    ↓ [conv4_x: Bottleneck ×6, stride 2, 채널 256→1024]
(1, 1024, 14, 14)     # 28÷2=14, 채널 256×4=1024
    ↓ [conv5_x: Bottleneck ×3, stride 2, 채널 512→2048]
(1, 2048, 7, 7)       # 14÷2=7, 채널 512×4=2048
    ↓ [Global Average Pooling]
(1, 2048)             # 7×7 특징 맵을 평균 내어 압축
    ↓ [FC Layer: 2048 → 1000]
(1, 1000)             # 1000개 클래스 점수
    ↓ [Softmax]
(1, 1000)             # 합이 1인 확률값 분포
```

**핵심 패턴 정리:**
- 이미지 크기(H, W)가 절반이 될 때마다 채널 수는 두 배가 됩니다. (계산량을 일정하게 유지하는 설계 원칙)
- 병목 블록에서 채널은 항상 **"압축 → 처리 → 복원"** 패턴을 따릅니다. (예: 256 → 64 → 64 → 256)

---

## 7. 면접 예상 질문 5개

<details>
<summary><b>Q1. ResNet이 해결하려 한 핵심 문제가 무엇인가요?</b></summary>

**힌트:** 더 깊은 네트워크가 더 얕은 네트워크보다 학습 오류가 높아지는 현상의 이름과 원인을 생각해보세요. 그리고 이것이 기울기 소실 문제와 어떻게 다른지도 떠올려 보세요.

</details>

<details>
<summary><b>Q2. Residual(잔차) 학습의 핵심 아이디어를 수식 없이 설명해보세요.</b></summary>

**힌트:** "정답을 직접 맞추는 것"과 "현재 답에서 얼마나 수정해야 하는가를 맞추는 것" 중 어떤 게 더 쉬울지 생각해보세요. 그리고 그 "수정분"이 0에 가까워지면 어떤 의미인지도 생각해보세요.

</details>

<details>
<summary><b>Q3. Shortcut connection이 역전파(backpropagation)에서 어떤 이점을 가져오나요?</b></summary>

**힌트:** 역전파(backpropagation)는 출력에서 입력 방향으로 기울기를 전달합니다. 숏컷 연결이 있으면 기울기가 어떤 "추가 경로"로 흐를 수 있을지 생각해보세요.

</details>

<details>
<summary><b>Q4. Bottleneck Block에서 1×1 conv를 사용하는 이유가 무엇인가요?</b></summary>

**힌트:** 1×1 conv는 공간적 정보를 바꾸지 않고 채널 수만 줄이거나 늘릴 수 있습니다. 3×3 conv의 입력 채널을 줄이면 계산량에 어떤 변화가 생기는지 생각해보세요.

</details>

<details>
<summary><b>Q5. ResNet-50과 ResNet-34의 구조적 차이점은 무엇이고, 왜 다른 블록 설계를 사용하나요?</b></summary>

**힌트:** ResNet-34는 Basic Block(3×3 conv 2개), ResNet-50은 Bottleneck Block(1×1 + 3×3 + 1×1 conv 3개)을 사용합니다. 두 블록의 계산량과 표현력 차이를 비교해보세요.

</details>

---

## 참고 자료

- 📄 원문 논문: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- 🔧 PyTorch 공식 구현: [torchvision.models.resnet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
