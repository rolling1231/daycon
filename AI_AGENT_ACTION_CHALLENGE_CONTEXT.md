# AI Agent Action Prediction Challenge Context

이 문서는 새 채팅 창에서도 AI 부문 프로젝트를 바로 이어갈 수 있도록 고정 배경 정보를 정리한 파일이다. 앞으로 이 프로젝트 관련 대화에서는 아래 내용을 기본 전제로 둔다.

## 주제

AI Agent 행동(Action) 의사결정 예측 챌린지

## 문제 설명

AI 코딩 에이전트 세션의 특정 시점에 기록된 상태 데이터를 바탕으로, 에이전트가 다음에 수행할 행동(action)을 14개 클래스 중 하나로 예측한다.

각 샘플은 다음 정보로 구성된다.

- `current_prompt`: 현재 사용자 발화
- `history`: 직전까지의 대화 및 행동 이력
- `session_meta`: 요금제, 잔여 토큰 예산, 작업공간 상태 등 세션 메타정보

예측 대상은 파일 읽기, 검색, 수정, 셸 명령 실행, 테스트 실행, 사용자 질문 등 에이전트의 주요 탐색, 수정, 실행, 대화 행동이다.

평가지표는 14개 클래스에 대한 `Macro-F1`이다.

## 제출 방식

`submit.zip` 파일을 제출한다.

필수 구조:

```text
submit.zip
├── model/
│   └── 예: model.pt, model.pkl 등
├── script.py
└── requirements.txt
```

평가 서버에서는 제출 파일에 아래 항목이 자동으로 추가된다.

```text
submit.zip
├── model/
├── script.py
├── requirements.txt
├── data/
└── output/
    └── submission.csv
```

`script.py`는 평가 서버에서 자동 실행되며, `data/` 디렉터리의 테스트 데이터를 읽고 `output/submission.csv` 파일을 반드시 생성해야 한다.

## 주요 제약

- 추론 코드 실행 시간: 10분 이하
- 패키지 설치 시간: 10분 이하
- 제출 zip 용량: 1GB 이하
- 실행 환경은 오프라인
- 패키지 설치 외 인터넷 연결 불가
- 사용 가능 언어: Python
- 평가 서버:
  - OS: Ubuntu 22.04.5 LTS
  - GPU: NVIDIA T4, 16GB VRAM
  - CPU: 3 vCPU
  - RAM: 12GB
  - Python: 3.11.15
  - CUDA: 12.8

## 데이터 구조

배포 데이터:

```text
open.zip
├── baseline_submit.zip
└── data/
    ├── train.jsonl
    ├── train_labels.csv
    ├── test.jsonl
    └── sample_submission.csv
```

파일 설명:

- `train.jsonl`: 학습 입력 데이터, 70,000건
- `train_labels.csv`: 학습 정답 데이터, 70,000행 x 2컬럼
- `test.jsonl`: 평가 입력 데이터 형식 확인용 5건 샘플
- `sample_submission.csv`: 제출 양식
- 실제 평가 시 비공개 테스트 데이터 30,000건이 평가 서버의 `data/test.jsonl`에 제공됨

## train.jsonl / test.jsonl 형식

각 줄은 JSON 객체 1개이며, 하나의 에이전트 세션 특정 시점 상태를 의미한다.

주요 필드:

- `id`: 샘플 고유 식별자
- `session_meta`: 세션 및 작업공간 메타정보
- `history`: 이전까지의 대화 및 행동 기록
- `current_prompt`: 현재 사용자 발화

예시 id:

```text
sess_sim_20260522_028750-step_02
```

### session_meta

`session_meta` 구성:

- `user_tier`: 요금제
  - `enterprise`
  - `pro`
  - `free`
- `language_pref`: 선호 언어
  - `ko`
  - `en`
  - `mixed`
- `budget_tokens_remaining`: 잔여 토큰 예산, 정수
- `turn_index`: 현재 턴 번호, 작을수록 세션 초반
- `elapsed_session_sec`: 세션 경과 시간, 초 단위
- `workspace`: 작업공간 상태

`workspace` 구성:

- `language_mix`: 코드베이스 언어 비율
  - 예: `{"py": 0.45, "sql": 0.30}`
  - 합은 대략 1.0
- `loc`: 전체 코드 라인 수
- `git_dirty`: 미커밋 변경 존재 여부, boolean
- `open_files`: 열려 있는 파일 경로 목록, 없으면 `[]`
- `last_ci_status`: 마지막 CI 상태
  - `passed`
  - `failed`
  - `none`

### history

`history`는 시간순 기록이며 0~12개 항목을 가진다.

사용자 턴:

- `role`
- `content`

행동 턴:

- `role`
- `name`: 행동명, 14개 클래스 중 하나
- `args`: 행동별 인자
- `result_summary`: 결과 요약

### current_prompt

현재, 즉 가장 최근 사용자 발화이다. 이 시점에서의 다음 행동이 예측 대상이다.

## train_labels.csv 형식

컬럼:

- `id`: `train.jsonl`과 연결되는 샘플 식별자
- `action`: 예측 대상 정답 클래스

## 예측 클래스 14개

모델은 아래 클래스명 중 정확히 하나를 예측해야 한다. 대소문자와 문자열이 완전히 일치해야 한다.

```text
read_file
grep_search
list_directory
glob_pattern
edit_file
write_file
apply_patch
run_bash
run_tests
lint_or_typecheck
ask_user
plan_task
web_search
respond_only
```

의미:

- `read_file`: 파일 읽기
- `grep_search`: 패턴 검색
- `list_directory`: 디렉터리 목록 확인
- `glob_pattern`: 글롭 패턴 검색
- `edit_file`: 기존 파일 수정
- `write_file`: 새 파일 작성
- `apply_patch`: 패치(diff) 적용
- `run_bash`: 셸 명령 실행
- `run_tests`: 테스트 실행
- `lint_or_typecheck`: 린트 또는 타입 검사
- `ask_user`: 사용자에게 질문
- `plan_task`: 작업 계획 수립
- `web_search`: 웹 검색
- `respond_only`: 도구 없이 응답만

## sample_submission.csv 형식

컬럼:

- `id`: 테스트 샘플 식별자
- `action`: 예측 클래스

`script.py`는 최종적으로 아래 경로에 같은 형식의 파일을 생성해야 한다.

```text
output/submission.csv
```

## 평가 서버 기본 설치 패키지

아래 패키지는 평가 서버에 기본 설치되어 있으므로, 가급적 `requirements.txt`에 다시 명시하지 않는다.

주요 Python 패키지:

```text
torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
narwhals==2.21.2
transformers==4.46.3
accelerate==1.9.0
sentencepiece==0.1.99
regex==2023.12.25
tqdm==4.66.4
loguru==0.7.2
pyyaml==6.0.1
rich==13.7.1
```

주요 시스템 패키지:

```text
git
build-essential
python3.11
python3.11-dev
python3.11-venv
python3-pip
libffi-dev
libblas3
liblapack3
libomp-dev
tzdata
unzip
p7zip-full
gfortran
libatlas-base-dev
default-jre-headless
cmake
pkg-config
ninja-build
libgl1
libglib2.0-0
```

## 프로젝트 운영 전제

- 목표는 `Macro-F1`을 높이는 것이다.
- 실제 테스트 데이터는 비공개 30,000건이며, 배포본 `test.jsonl` 5건은 형식 확인용이다.
- 오프라인 실행이므로 외부 모델 다운로드, API 호출, 웹 검색 의존 추론은 불가능하다.
- 제출물은 `script.py`, `requirements.txt`, `model/`만으로 재현 가능해야 한다.
- 기본 설치 패키지를 최대한 활용해 설치 시간을 줄인다.
- T4 GPU가 있지만, 10분 추론 제한과 12GB RAM을 고려해 모델 크기와 배치 전략을 보수적으로 잡아야 한다.
- `data/`는 읽기 전용이고, 결과는 `output/submission.csv`에 저장해야 한다.
