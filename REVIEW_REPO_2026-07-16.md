# Review Toan Bo Repo - 2026-07-16

## Pham vi review

Review nay duoc thuc hien theo huong "code review + repo audit", uu tien:

- Tinh dung cua luong chay chinh (`COLMAP -> train -> render -> eval -> package`).
- Do nhat quan giua code hien tai va tai lieu huong dan.
- Rui ro van hanh, kha nang tai lap, cac diem de gay sai so am tham.
- Kha nang su dung thuc te cua cac script phu tro phan tich/chan doan.

Da doc truc tiep cac file code chinh trong `pipeline/common/` va `pipeline/scripts/`, cac README/plan/worklog o root, va quet nhanh noi dung notebook/de tail van ban de tim dau hieu drift giua round 1 va round 2.

Khong chay thu pipeline that, khong train, khong render, khong sua ma nguon. Moi nhan dinh ben duoi dua tren static review.

## Tong ket nhanh

Repo co chat luong ky thuat phan code loi kha tot so voi mat bang do an/nghien cuu:

- Docstring rat day, giai thich ro convention pose/COLMAP/3DGS.
- Nhieu assertion/canh bao de chan sai so am tham.
- Logging tach file hop ly.
- Luong package submission da co verify dinh dang/kich thuoc, day la diem manh ro ret.
- Co xu ly rat thuc dung cho cac bai hoc "da tung gap that" nhu antialiasing drift, sparse co anh thua, cleanup dia, depth map 16-bit.

Van de lon nhat cua repo hien tai khong nam o logic loi, ma nam o **drift boi canh**:

- Code loi da chuyen sang **round 2**.
- Mot luong lon tai lieu, notebook va entrypoint van noi theo **round 1**.
- Ket qua la nguoi dung moi rat de chay sai notebook, sai dataset, sai lenh CLI, hoac tin nham script phan tich da "con dung" trong khi thuc te da lech boi canh.

## Phat hien chinh

### 1. Nghiem trong: Tai lieu chinh cua repo da lech boi canh so voi code hien tai, de dan den chay sai pipeline ngay tu diem vao

**Bang chung**

- `README.md:1-29` van mo ta repo la "Vong 1", dataset `VAI_NVS_DATA`, deadline `submission_round1.zip`, va tro den notebook `pipeline/kaggle_pipeline.ipynb` khong ton tai.
- `pipeline/README.md:1-125` van huong dan theo 13 scene, `public_set/private_set1`, dung cac lenh `--split public`, `--split private`, trong khi:
  - `pipeline/common/scenes.py:1-34` da chuyen sang `VAI_NVS_DATA_ROUND2`, 7 scene phang, khong con split.
  - `pipeline/scripts/01_run_colmap.py:95-121` khong co tham so `--split`, ma dung `--domain`.
  - `pipeline/scripts/06_package_submission.py:158-177` package cho toan bo 7 scene round 2.

**Tac dong**

- Nguoi doc README root se di den file notebook sai/khong ton tai.
- Nguoi lam theo `pipeline/README.md` se gap lenh CLI khong hop le ngay o buoc 1.
- Rui ro cao hon la chay nham notebook/tai nham dataset round 1 trong khi code loi dang doc round 2.
- Day la loai loi "entrypoint/documentation breakage": code co the dung, nhung nguoi dung khong vao duoc dung duong.

**Danh gia**

- Muc do: `High`
- Ly do xep muc do cao: no anh huong truc tiep den kha nang su dung repo, khong can cho den luc train moi vo.

### 2. Trung binh-cao: `09_diagnose_distance.py` khong con khop voi quy trinh danh gia holdout cua round 2, nen de phan tich sai hoac khong dung duoc

**Bang chung**

- `pipeline/scripts/09_diagnose_distance.py:16-22` van mo ta cach dung cho "scene public".
- `pipeline/scripts/09_diagnose_distance.py:87-99` luon map `eval_metrics.csv` voi `scene.test_poses_csv`.
- Trong khi do, quy trinh round 2 hien tai cua `pipeline/scripts/05_eval_metrics.py:7-19` va `:177-192` danh gia tren **holdout**; file `eval_metrics.csv` duoc sinh tu `holdout_gt` va `holdout_renders`, tuc stem anh thuoc `holdout_poses.csv`, khong phai `test_poses.csv` that cua scene.

**He qua ky thuat**

- Neu `eval_metrics.csv` la ket qua holdout dung theo flow hien tai, `09_diagnose_distance.py` se co xac suat cao khong map duoc stem -> pose:
  - in nhieu dong `[BỎ QUA]`.
  - cuoi cung `len(rows) < 5` va exit.
- Neu co mot it stem vo tinh trung nhau, script van co the cho ra phan tich nhung tren tap khong day du, rat de dan den ket luan sai.

**Danh gia**

- Muc do: `Medium-High`
- Ly do: day khong pha luong train/render/package chinh, nhung pha huong phan tich ra quyet dinh. Trong repo nghien cuu, day la mot hu hong quan trong.

### 3. Trung binh: `05_eval_metrics.py` cho phep bo qua LPIPS nhung cach tinh Score khi thieu LPIPS de gay hieu nham

**Bang chung**

- `pipeline/scripts/05_eval_metrics.py:102-107`:
  - comment noi "coi nhu 0 cho phan (1 - LPIPS)".
  - code dat `lpips_term = 0.0 if np.isnan(lpips_v) else (1.0 - lpips_v)`.
- `pipeline/scripts/05_eval_metrics.py:166-173` cho phep chay tiep khi khong co package `lpips`.

**Van de**

- Ve mat toan hoc, khi khong co LPIPS ma dat `lpips_term = 0.0`, ban da xoa toan bo 40% thanh phan dau tien cua score.
- Ket qua Score bi keo xuong manh va co he thong.
- Nguoi dung co the so sanh A/B giua cac run dua tren score nay va rut ra ket luan sai, dac biet khi chenh lech that giua hai run nho.

**Luu y**

- Docstring da co canh bao "KHONG dai dien dung diem that", nen day nghieng nhieu ve van de thiet ke/UX hon la bug logic thuan tuy.
- Tuy nhien, vi script van in ra "Score mean" va "uoc luong leaderboard", muc do gay hieu nham van dang dang ke.

**Danh gia**

- Muc do: `Medium`

### 4. Trung binh: Notebook va tai lieu phu van con nhieu noi dung round 1, tao "dual reality" trong repo

**Bang chung**

- Quet noi dung cho thay:
  - `pipeline/kaggle_public.ipynb` van noi `public_set/private_set1`, `phase1`, scene public cua round 1.
  - `README.md` va `pipeline/README.md` cung con noi theo round 1.
- Trong khi code loi:
  - `pipeline/common/scenes.py:1-34` va cac script chinh da su dung round 2.

**Tac dong**

- Repo hien tai ton tai dong thoi hai "su that":
  - su that trong code loi.
  - su that trong notebook/README cu.
- Day la nguon gay nham lan lon nhat khi handoff cho nguoi khac, hoac khi chinh tac gia quay lai repo sau mot thoi gian.

**Danh gia**

- Muc do: `Medium`
- Day chu yeu la no ky thuat van hanh va quan ly tri thuc, nhung no anh huong truc tiep den kha nang tai lap.

## Danh gia kien truc va chat luong code

### 1. Nhung diem lam tot

#### 1.1. `common/poses.py` duoc viet can than

- Docstring ro quy uoc `world -> camera`, `qvec/tvec`, `R/T/FOV`.
- Co check `assert_centered_principal_point()` thay vi bo qua am tham.
- Co tach rieng `representative_intrinsics()` de tai su dung cho prior COLMAP.

Nhan xet: day la file "nen tang", va duoc viet dung tinh than phong tranh sai convention, rat quan trong voi NVS/COLMAP.

#### 1.2. `common/colmap_runner.py` la mot trong nhung file manh nhat repo

- Co xu ly truong hop sparse cua BTC chua anh ma tren dia khong con file that.
- Chu dong loc reconstruction truoc khi `undistort_images`, tranh crash muon trong `train.py`.
- Co verify lai sparse sau undistort de dam bao khong con anh "mo coi".
- Co logging ro va tom tat console gon.

Nhan xet: day la kieu code rat thuc te, duoc viet sau khi da gap case that. Gia tri ky thuat cao hon nhieu so voi baseline "goi pycolmap thang".

#### 1.3. Dong bo flag train/render da duoc xu ly dung huong

- `pipeline/scripts/03_train_3dgs.sh` ghi them `pipeline_train_flags.json`.
- `pipeline/scripts/04_render_test_poses.py` doc lai file nay de khop `antialiasing`.
- `pipeline/scripts/10_sanity_check_render.py` tiep tuc kiem chung duong render submission bang anh train that.

Nhan xet: day la mot cum giai phap rat hop ly cho loi "train dung mot cau hinh, render bang cau hinh khac". Ve ky thuat, day la diem cong lon.

#### 1.4. `06_package_submission.py` duoc lam chac tay

- Check du so anh.
- Check dung kich thuoc.
- Encode lai dung theo duoi file trong zip.
- Verify lai chinh zip vua tao.

Nhan xet: trong bai thi kieu nay, package script thuong la noi vo vao phut cuoi. O day script nay lai la mot trong nhung phan on nhat repo.

### 2. Cau truc repo hien tai

Repo thuc chat gom 4 tang:

- Tang boi canh:
  - `README.md`, `Đề bài.md`, `Hướng đi.md`, `plan.md`, `WORKLOG.md`.
- Tang code loi:
  - `pipeline/common/*.py`
  - `pipeline/scripts/01..11`
- Tang notebook van hanh:
  - cac file `pipeline/*.ipynb`
- Tang artifact/ket qua:
  - `Kết quả/`

Van de khong nam o viec tach tang, ma o cho tang boi canh va tang notebook khong con dong bo voi tang code loi.

### 3. Muc do gan ket/thiet ke

Repo thien ve "script-driven research pipeline", khong phai package hoa theo kieu san pham. Dieu nay chap nhan duoc voi do an/nghien cuu, nhung co 3 he qua:

- Phu thuoc manh vao dung thu tu thao tac.
- Phu thuoc manh vao docstring/noi dung text de hieu "tai sao".
- Drift tai lieu se nguy hiem hon mot codebase package hoa.

Trong boi canh nay, viec duy tri 1 "source of truth" duy nhat la rat quan trong. Hien tai repo chua co.

## Danh gia tung cum script

### `00_make_holdout_split.py`

Diem tot:

- Cach chon holdout trai deu theo trajectory la hop ly hon random don thuan.
- Co giai thich ro ve quy uoc quaternion cua pycolmap.
- Tai su dung duoc machinery loc sparse da co san trong `colmap_runner.py`.

Rui ro nho:

- Script duoc thiet ke theo gia dinh cau truc output la symlink phang; neu sau nay thu muc holdout them subdir/phu tro khac, logic `unlink()` tung file se mong manh hon.
- Day hien chua la bug, chi la mot diem can de y neu mo rong.

### `01_run_colmap.py`

Diem tot:

- API don gian.
- Co `--holdout`.
- Co fallback `--force_own_colmap`.
- Messaging nguoi dung kha ro.

Van de:

- Docstring dau file van con cau "13/13 scene" (`pipeline/scripts/01_run_colmap.py:5-8`) trong khi `common/scenes.py` da la round 2, 7 scene. Khong gay sai logic, nhung cho thay dau vet drift tai lieu ngay trong script.

### `02_validate_frame.py`

Diem tot:

- Muc tieu rat ro: script kiem dinh, khong lam nhieu viec hon can thiet.
- So sanh bang camera centers + Sim3 la huong dung.

Van de:

- Phan thong diep ket luan cuoi file van con dau vet boi canh round 1/private scenes, nen nguoi doc de bi nham muc dich that cua script o round 2.

### `03_train_3dgs.sh`

Diem tot:

- Rat giau kinh nghiem van hanh that: canh bao OOM, free disk, checkpoint giua chung, cleanup, antialiasing/depth/exposure gating.
- Co y thuc ro ve commit pin cua repo ngoai.

Rui ro:

- Script dang gan kha nhieu tri thuc nghiep vu va operational policy vao shell script duy nhat. Ve sau neu repo phat trien tiep, day co the thanh "god script".
- Hien tai van chap nhan duoc vi pham vi con hep.

### `04_render_test_poses.py`

Diem tot:

- Tu dung camera cho pose tuy y la dung bai toan.
- Co check kich thuoc output.
- Co luu log tung anh.

Rui ro:

- Duong render nay la duong code song song voi duong camera khi train, nen phu thuoc rat manh vao `10_sanity_check_render.py` de giu tinh dung. May man la repo da co script sanity nay.

### `05_eval_metrics.py`

Diem tot:

- Co giai thich ro score formula.
- Co gom diem theo scene dung huong.
- Co in do nhay theo `PSNR_max`.

Van de da neu o muc Phat hien 3:

- Score khi thieu LPIPS de gay hieu nham.

### `06_package_submission.py`

Danh gia:

- Day la script dang tin cay nhat trong nhom "script cuoi duong ong".
- It mui ky thuat, guardrails day du.

### `07_build_antenna_weights.py` va `apply_antenna_patch.py`

Diem tot:

- Y tuong nhat quan voi domain BTS.
- Co gioi han pham vi cho scene `bts`, khong lam bua.

Rui ro:

- `apply_antenna_patch.py` phu thuoc thay the theo text nguyen khoi. Tac gia da biet dieu nay va docstring da canh bao ro.
- Nghia la day la huong nghien cuu hop le, nhung maintainability yeu. Neu muon bien thanh thanh phan chinh thuc, can doi sang patch co test hoac fork ro rang.

### `08_generate_depth_priors.py`

Diem tot:

- Nhinh ra duoc bug 8-bit/16-bit o Depth Anything V2 README flow la rat gia tri.
- Cach tach `generate_depth_maps()` va `run_make_depth_scale()` ro.

Rui ro:

- Phu thuoc vao hai repo ngoai cung luc (`DA_REPO`, `GS_REPO`), nen kha nang vo moi truong cao. Tuy nhien script da canh bao tuong doi ro.

### `09_diagnose_distance.py`

Danh gia:

- Y tuong phan tich tot.
- Hien tai boi canh su dung da lech, xem Phat hien 2.

### `10_sanity_check_render.py`

Danh gia:

- Rat nen giu.
- Day la "safety net" co gia tri cao nhat repo sau package verify.
- Trong pipeline nghien cuu, script nay giup phan biet "model te that" va "render path sai".

### `11_trr_refine.py`

Diem tot:

- Co y thuc ro ve gioi han phuong phap.
- Ghi chu rat ky ve van de `points2D.xy` scale mismatch.
- Tu choi hallucinate, giu huong an toan.

Rui ro:

- Script phuc tap hon nhung script con lai, phu thuoc manh vao nhieu gia dinh hinh hoc ngầm.
- Chua thay test nao cho nhom logic nay; voi script nghien cuu thi chap nhan duoc, nhung can xem no la "experimental", khong phai thanh phan core da on dinh.

## Rui ro he thong

### 1. Khong co "nguon su that duy nhat"

Hien tai nguoi doc co the lay thong tin tu:

- `README.md`
- `pipeline/README.md`
- `plan.md`
- `WORKLOG.md`
- notebook
- docstring trong script

Nhung cac nguon nay khong dong bo. Day la rui ro he thong so 1.

### 2. Phu thuoc cao vao tri nho tac gia

Repo ghi lai rat nhieu kinh nghiem quy gia, nhung chung dang nam rai rac:

- mot phan trong docstring
- mot phan trong `WORKLOG.md`
- mot phan trong `plan.md`
- mot phan trong ten bien moi truong/CLI

Neu ban giao cho nguoi khac, chi phi hoc lai boi canh se lon.

### 3. Moi truong chay kho tai lap neu khong theo sat tai lieu dung

Ban than code co y thuc pin commit va kiem soat bien moi truong kha tot. Van de la:

- tai lieu dang tro sai noi
- notebook co noi dung cu

Nen "reproducibility in code" tot hon "reproducibility in repo".

## Muc do san sang su dung

Neu danh gia theo 2 lop:

- **Lop code loi round 2**: kha san sang de tiep tuc dung nghien cuu/thuc thi.
- **Lop repo nhu mot san pham co the handoff**: chua san sang, vi tai lieu va entrypoint dang gay nham lan.

Noi cach khac:

- Neu chinh tac gia repo quay lai va nho boi canh: van lam viec duoc.
- Neu mot nguoi khac clone repo va lam theo README: kha nang cao se gap sai ngay som.

## Uu tien khuyen nghi

### Uu tien 1

Hop nhat boi canh round 2 tai 3 diem vao chinh:

- `README.md`
- `pipeline/README.md`
- notebook nao duoc coi la entrypoint chinh

Muc tieu:

- Chi de ton tai 1 cach "bat dau dung".
- Loai bo hoan toan lenh/duong dan/notebook round 1 khoi README chinh.

### Uu tien 2

Sua hoac danh dau ro `09_diagnose_distance.py` la:

- chi dung cho round 1/public-set, hoac
- nang cap de nhan `--poses_csv`/`holdout_poses.csv`.

Neu khong, nen xem script nay dang "phan nao hong theo boi canh hien tai".

### Uu tien 3

Lam ro hanh vi cua `05_eval_metrics.py` khi khong co LPIPS:

- hoac cam tinh "Score" neu thieu LPIPS,
- hoac in thong diep canh bao manh hon rang score do khong dung de so A/B.

### Uu tien 4

Chon 1 tai lieu lam "single source of truth" cho round 2, vi du:

- `plan.md` neu muc tieu la nghien cuu,
- hoac `pipeline/README.md` neu muc tieu la van hanh.

Con cac file khac chi nen tro link den tai lieu nay, khong lap lai huong dan day du nua.

## Ket luan

Danh gia tong the:

- **Code loi**: tot, can than, co nhieu bai hoc thuc chien, kha dang tin.
- **Repo o goc do handoff/operability**: yeu vi drift tai lieu va notebook.

Neu chi duoc chot 1 cau:

> Bai toan lon nhat cua repo nay hien tai khong phai "code co dung khong", ma la "nguoi dung se tin file nao la dung".

Do do, neu muon repo tiep tuc co gia tri trong nhung lan chay sau, buoc quan trong nhat khong phai viet them thuat toan moi, ma la dong bo lai boi canh round 2 tren tat ca diem vao.

## Goc nhin dieu hanh chuyen nghiep: To chuc theo AI Agent de toi da hoa chat luong model

Phan nay duoc viet lai theo goc nhin van hanh kieu doanh nghiep nho hoac phong R&D thi dau:

- Moi cong doan co owner ro rang.
- Moi Agent co KPI, input, output, va dieu kien ban giao.
- Moi quyet dinh rollout deu di qua gate, khong dua vao cam tinh.
- Muc tieu khong phai "thu nhieu y tuong", ma la **ra quy trinh san xuat ket qua tot nhat voi rui ro thap nhat**.

## Muc tieu kinh doanh ky thuat

Neu xem bai thi la mot bai toan kinh doanh toi uu hoa tai nguyen, muc tieu dung phai la:

- Toi da hoa score cuoi cung tren submission.
- Toi thieu hoa GPU-gio bi dot vao thu nghiem vo nghia.
- Toi thieu hoa rui ro "nhin thay model tot nhung thuc ra pipeline do sai".
- Toi thieu hoa rui ro "24 gio cuoi moi phat hien package sai / render sai / config drift".

Noi ngan gon:

> He thong can duoc van hanh nhu mot day chuyen san xuat model, khong phai mot bo script thu cong.

## Co cau AI Agent de van hanh du an

Toi khuyen nghi chia thanh 7 Agent chuc nang. Co the la 7 prompt/7 session/7 role rieng, hoac 3-4 Agent chinh + 1 nguoi tong hop. Van de khong nam o so luong Agent, ma o **tach vai tro ro rang**.

### Agent 1: Program Manager

Vai tro:

- Dieu phoi toan bo pipeline.
- Chot thu tu thu nghiem.
- Phe duyet viec rollout mot ky thuat moi.
- Quan ly tai nguyen GPU, thoi gian, va rui ro deadline.

Input:

- Bao cao tu cac Agent con lai.
- Bang score holdout theo scene/domain.
- Tinh trang artifact: model, render, zip, log.

Output:

- 1 backlog uu tien ro rang.
- 1 lich chay GPU theo ngay/ca.
- 1 quyet dinh cuoi cung: giu baseline hay rollout ky thuat moi.

KPI:

- So thu nghiem vo nghia bi cat bo.
- Ty le thu nghiem co ket luan ro rang.
- Khong de 2 Agent cung nghien cuu trung mot gia thuyet ma khong biet.

Quyen han:

- Co quyen dong băng bat ky huong nghien cuu nao khong du bang chung.
- Co quyen bat buoc quay ve baseline neu quy trinh do dang mat kiem soat.

### Agent 2: Baseline Integrity Agent

Vai tro:

- Bao dam baseline chay dung, do dung, lap lai duoc.
- Day la Agent quan trong nhat trong giai doan dau.

Nhiem vu:

- Khoa holdout split co dinh.
- Xac nhan `01_run_colmap.py -> 03_train_3dgs.sh -> 04_render_test_poses.py -> 05_eval_metrics.py` la mot luong do hop le.
- Bat buoc chay `10_sanity_check_render.py`.
- Kiem tra train/render co khop `antialiasing`, `sh_degree`, va artifact model hay khong.

Input:

- Dataset round 2.
- Config baseline.
- Log train/render/eval.

Output:

- 1 baseline benchmark da duoc xac nhan hop le.
- 1 bang "known-good configuration".
- 1 danh sach loi van hanh da loai bo.

KPI:

- Baseline chay lap lai tren cung holdout cho sai khac nho.
- Khong con false improvement do config drift.
- Khong con run nao "khong ro co hop le hay khong".

Quy tac:

- Neu baseline chua on dinh, cam mo them huong depth/TRR/antenna-focus.

### Agent 3: Data & Scene Strategy Agent

Vai tro:

- Khong coi 7 scene la mot khoi dong nhat.
- Tim cach chia nhom scene de dat config theo domain.

Nhiem vu:

- Xac nhan nhom `bts` va `generic`.
- Chon scene benchmark dai dien:
  - 1 scene BTS kho.
  - 1 scene BTS trung binh.
  - 1 scene generic.
- Ghi nhan dac diem cua tung nhom:
  - nhieu day/cot/anten
  - nhieu mat lap lai
  - scene generic de render on dinh hon hay khong

Input:

- `pipeline/common/scenes.py`
- Ket qua holdout
- Anh render va metric tung scene

Output:

- 1 ma tran scene theo domain va do kho.
- 1 khuyen nghi config theo domain.

KPI:

- So lan tranh duoc sai lam "1 config cho tat ca".
- Muc do giam so thu nghiem khong can thiet tren 7 scene.

Quan diem:

- O repo nay, chia domain la mot co hoi chien luoc lon.
- Day co kha nang mang lai ROI cao hon viec vat tung tham so nho tren toan bo 7 scene.

### Agent 4: Geometry Improvement Agent

Vai tro:

- Chiu trach nhiem cho cac cai tien tac dong vao geometry that cua scene.
- Day la Agent uu tien thu nghiem truoc Agent hau xu ly.

Pham vi:

- Depth prior.
- Chon config train co tac dong den kha nang tai tao hinh hoc.
- Kiem tra floaters / vung thieu support nhin tu camera train.

Nhiem vu chinh:

- Thu nghiem `DEPTH_PRIOR=1`.
- So sanh baseline vs depth prior tren benchmark scenes.
- Tra loi:
  - BTS co duoc loi hon generic khong.
  - Loi nam o PSNR, SSIM, LPIPS hay tat ca.
  - Co side effect over-smooth hoac lam mat texture khong.

Input:

- Baseline da khoa.
- Depth maps 16-bit.
- Eval metrics + inspection hinh.

Output:

- Bao cao A/B dung chuan.
- De xuat rollout:
  - rollout toan bo
  - rollout theo domain
  - khong rollout

KPI:

- Muc tang score thuc chat tren holdout.
- Do on dinh giua cac run.
- Ty le scene duoc loi sau rollout.

Nhan dinh chuyen mon:

- Neu chi duoc uu tien 1 huong tang diem trong repo nay, toi uu tien Agent nay truoc tien.

### Agent 5: Rendering & Consistency Agent

Vai tro:

- Bao dam duong render de nop bai phan anh dung model da train.
- Day la Agent "bao hiem chat luong".

Nhiem vu:

- Kiem tra `04_render_test_poses.py`.
- Kiem tra `pipeline_train_flags.json`.
- Chay `10_sanity_check_render.py` tren cac run benchmark va run final.
- Kiem tra kich thuoc anh, principal point, iteration render.

Input:

- Model checkpoint.
- Sparse/pose CSV.
- Render output.

Output:

- Xac nhan run hop le de dua vao bang score.
- Loai bo cac run score cao/score thap do render mismatch.

KPI:

- So loi render mismatch bi bat som.
- Khong de run final moi phat hien train/render lech config.

Nguyen tac:

- Run nao khong qua gate consistency thi xem nhu vo hieu, khong dua vao quyet dinh.

### Agent 6: Post-processing & Booster Agent

Vai tro:

- Chiu trach nhiem cho cac huong tang diem "sau khi baseline geometry da on".

Pham vi:

- `11_trr_refine.py`
- Moi hau xu ly reference-guided an toan
- Co the ve sau moi dong vao antenna-focus neu can

Nhiem vu:

- Chi bat dau sau khi Program Manager xac nhan baseline + geometry da on.
- Thu TRR tren 1-2 scene benchmark.
- Bao cao:
  - coverage
  - artifact seam
  - metric tang that hay khong

Input:

- Render baseline tot.
- Sparse + train images.
- Holdout/test poses.

Output:

- Ket luan "TRR co xung dang rollout hay khong".
- Neu rollout, rollout cho domain nao.

KPI:

- Muc tang them tren config da on dinh.
- Khong pha hinh vung tot de cuu vung xau.

Lap truong:

- Day la booster.
- Khong duoc phep dong vai "thuoc chua benh" cho mot baseline con sai geometry.

### Agent 7: Submission Reliability Agent

Vai tro:

- Quan ly cong doan cuoi cung nhu QA cua doanh nghiep.

Nhiem vu:

- Kiem tra render du scene/du anh.
- Chay package.
- Verify zip.
- Kiem tra dung ten file, dung format, dung kich thuoc.
- Xac nhan run final nao duoc phep nop.

Input:

- Render final.
- `06_package_submission.py`
- Log verify.

Output:

- 1 submission zip da duoc xac nhan.
- 1 bien ban release: config nao da tao ra zip nay.

KPI:

- So lan package loi = 0 o giai doan cuoi.
- Moi file nop deu truy vet duoc ve model/config nguon.

## Mo hinh van hanh de toi da hoa ket qua

Toi khuyen nghi chia van hanh thanh 4 phase.

### Phase 1: Stabilize

Muc tieu:

- Tao baseline hop le va tai lap duoc.

Agent chinh:

- Program Manager
- Baseline Integrity Agent
- Rendering & Consistency Agent

Cong viec:

- Chon benchmark scenes.
- Khoa holdout.
- Chay baseline lap lai.
- Bat buoc sanity-check render.
- Dong bo mot "known-good runbook".

Tieu chi thoat phase:

- Baseline duoc xac nhan dung.
- Metric holdout co the so sanh A/B.
- Khong con ngh nghi ve config drift.

### Phase 2: Improve Geometry

Muc tieu:

- Tim cai tien co kha nang tang diem cao nhat.

Agent chinh:

- Geometry Improvement Agent
- Data & Scene Strategy Agent

Cong viec:

- Thu depth prior tren benchmark.
- So sanh theo domain.
- Chot chinh sach:
  - tat ca scene dung depth
  - chi BTS dung depth
  - bo qua depth

Tieu chi thoat phase:

- Co 1 config geometry duoc chon bang so lieu.

### Phase 3: Booster Layer

Muc tieu:

- Tim 1 lop tang diem bo sung sau khi geometry da on.

Agent chinh:

- Post-processing & Booster Agent
- Rendering & Consistency Agent

Cong viec:

- Thu TRR.
- Danh gia tac dong metric va artifact.
- Chi rollout neu co loi ro va khong gay regressions lon.

Tieu chi thoat phase:

- Co hoac khong co booster, nhung quyet dinh da duoc khoa.

### Phase 4: Finalization

Muc tieu:

- Bien toan bo ket qua thanh submission on dinh.

Agent chinh:

- Submission Reliability Agent
- Program Manager

Cong viec:

- Retrain final theo config da chot.
- Render full 7 scene.
- Package.
- Verify.
- Luu lai release note.

Tieu chi ket thuc:

- Co 1 artifact nop bai duy nhat, truy vet duoc, da verify.

## Decision Gate theo kieu doanh nghiep

Khong cho phep rollout "vi cam thay dep". Moi thay doi phai qua 1 gate.

### Gate 1: Validity Gate

Run chi hop le neu:

- Train xong hop le.
- Render xong hop le.
- `10_sanity_check_render.py` khong bao dau hieu bat thuong.
- Metric tinh tren dung holdout.

Fail gate nay:

- Loai run.
- Cam dua vao bang so sanh.

### Gate 2: Improvement Gate

Ky thuat moi chi duoc coi la co trien vong neu:

- Tang lap lai tren it nhat 2 benchmark scene, hoac
- Tang ro rang tren 1 domain va khong lam domain do te di o scene khac.

Khong chap nhan:

- Tang 1 run duy nhat.
- Tang chi vi config drift.
- Tang PSNR nhung vo LPIPS/SSIM nang ma khong co ly do chien luoc.

### Gate 3: Rollout Gate

Chi rollout toan bo khi:

- Program Manager co bao cao tong hop.
- Data & Scene Strategy Agent xac nhan pham vi rollout.
- Submission Reliability Agent xac nhan rollout khong lam vo quy trinh cuoi.

## Thu tu uu tien chien luoc toi khuyen nghi

### Uu tien cap 1: Bao toan score

Lam ngay:

- Khoa baseline.
- Khoa holdout.
- Khoa antialiasing thanh luat.
- Bat buoc sanity-check render.
- Dong bo entrypoint round 2 de tranh chay nham.

Ly do:

- Day la nhung thu giup tranh mat diem oan.
- Trong thi dau, mat diem oan nguy hiem hon viec chua tim duoc y tuong moi.

### Uu tien cap 2: Tang diem bang geometry

Lam sau khi cap 1 on:

- Depth prior.
- Chia config theo domain BTS/generic.

Ly do:

- Day la huong co ROI cao nhat dua tren boi canh repo.
- Tac dong vao geometry thuong ben vung hon hau xu ly.

### Uu tien cap 3: Tang diem bang booster

Lam khi geometry da on:

- TRR refine.

Ly do:

- Co the lay them diem.
- Nhung khong nen de no gay nhieu xao tron khi baseline chua on.

### Uu tien cap 4: Nghien cuu rui ro cao

Chi lam khi con tai nguyen:

- Antenna-focus patch trainer.
- Cac thay doi sau vao vong lap train.

Ly do:

- Co kha nang hay.
- Nhung kha nang an thoi gian va gay side effect rat cao.

## Ke hoach thuc thi cu the nhu mot doanh nghiep nho

### Sprint 1: Baseline Certification

Owner:

- Baseline Integrity Agent

Deliverable:

- 1 file benchmark baseline theo 2-3 scene.
- 1 known-good config.
- 1 bien ban xac nhan render consistency.

### Sprint 2: Geometry Expansion

Owner:

- Geometry Improvement Agent

Deliverable:

- Bao cao baseline vs depth prior.
- Khuyen nghi rollout theo domain.

### Sprint 3: Booster Validation

Owner:

- Post-processing & Booster Agent

Deliverable:

- Bao cao TRR co nen dung hay khong.
- Neu dung, dung cho domain nao.

### Sprint 4: Release Candidate

Owner:

- Submission Reliability Agent

Deliverable:

- 1 release candidate zip.
- 1 release note:
  - config
  - scenes
  - metric holdout
  - checksum/duong dan artifact

## Y kien chuyen mon thang tay

Neu muc tieu la ket qua model tot nhat trong thoi gian co han, toi khong to chuc du an theo kieu:

- Moi Agent di thu 1 y tuong rieng cho vui.
- Thich gi thu nay.
- Thay hinh dep hon la rollout.

Toi se to chuc theo kieu:

- 1 Agent bao ve baseline.
- 1 Agent chuyen gia geometry.
- 1 Agent chuyen gia booster.
- 1 Agent chuyen gia package/release.
- 1 Agent quan ly quyet dinh.

Va quy tac cao nhat la:

> Khong co Agent nao duoc phep tu minh "thang" bang ly le. Chi score hop le, lap lai duoc, qua gate moi duoc rollout.

## Ket luan chien luoc cuoi cung

Neu phai dua ra mot phuong an dieu hanh toi uu cho repo nay, toi chot nhu sau:

- To chuc he thong Agent theo vai tro ro rang.
- Khoa baseline va quy trinh do la uu tien so 1.
- Depth prior la huong cai tien chinh nen dau tu truoc.
- Chia config theo domain BTS/generic thay vi co gang tim 1 config cho tat ca.
- TRR chi la lop tang diem sau cung.
- Antenna-focus la huong R&D rui ro cao, de sau.
- Submission phai co 1 Agent QA rieng, khong giao chung cho Agent train model.

Noi ngan gon:

> Muon ra ket qua model tot nhat, repo nay can duoc van hanh nhu mot to chuc co gate va KPI, khong phai mot chuoi script thu cong. Huong kiem tien diem cao nhat hien tai la baseline sach -> depth prior -> rollout theo domain -> booster sau cung -> package nhu release production.

## De xuat 1 cai tien toi uu nhat trong dieu kien tai nguyen gioi han

Neu bat toi phai chon **chi 1 cai tien lon nhat, dang dau tu nhat, co xac suat tang diem thuc chat cao nhat** tren co so repo nay, toi se khong chon:

- them mot patch trainer phuc tap,
- them mot hau xu ly dep mat,
- hoac vat qua nhieu hyperparameter nho.

Toi se chon mot cai tien duy nhat sau:

> **Domain-Adaptive Depth-Guided Training**
>
> Tien hanh train theo 2 che do khac nhau cho 2 domain, trong do 5 scene BTS duoc bat depth prior nhu mot thanh phan chinh thuc cua baseline, con 2 scene generic giu baseline sach neu A/B cho thay depth khong on dinh.

Noi de hieu:

- Khong co "mot baseline cho tat ca scene".
- Khong co "depth prior la mot thu nghiem phu".
- Ta bien no thanh **kien truc rollout theo domain**.

Day la cai tien toi uu nhat vi no danh dung 3 diem yeu that cua bai toan:

- scene BTS co nhieu cau truc mong, lap lai, de sinh floaters;
- repo da co ha tang depth prior kha day du, nghia la chi phi trien khai thap hon cac y tuong moi;
- 2 scene generic co phan bo hinh hoc khac, nen khong nen ep cung mot regularization policy.

## Tai sao toi chon huong nay thay vi cac huong khac

### 1. No tac dong vao geometry, khong chi vao be mat anh

Trong bai toan NVS, neu geometry sai:

- floaters se xuat hien,
- vung day/cot/khung mong se vo,
- goc nhin moi se te hon rat nhanh.

Nhieu huong hau xu ly co the lam anh "dep hon", nhung khong sua goc geometry. Depth prior thi co kha nang tac dong vao:

- phan bo Gaussian,
- su on dinh cua reconstruction,
- kha nang giu cau truc mong khi render pose la.

Neu chi duoc dau tu 1 huong, huong sua geometry co gia tri dai han hon huong makeup ket qua.

### 2. No dung voi domain BTS hon cac y tuong tong quat

5 scene BTS khong giong `bonsai` va `chair`.

Dieu quan trong khong phai chi la "BTS co antenna", ma la:

- nhieu thanh kim loai mong;
- nhieu canh occlusion;
- nhieu vung lap lai texture;
- nhieu hinh hoc kho cho 3DGS vanilla.

Depth prior la cach hop ly nhat de dua vao mot rang buoc bo sung ma khong can viet lai toan bo trainer.

### 3. No co ROI cao nhat tren repo hien tai

Repo nay da san:

- `08_generate_depth_priors.py`
- gating trong `03_train_3dgs.sh`
- toan bo ly giai 16-bit va `depth_params.json`

Nghia la:

- y tuong da co nen tang ky thuat;
- chi phi implementation thap;
- xac suat gap bug "vi vua nghi ra vua code" thap hon anten-focus hay patch trainer moi.

Trong dieu kien tai nguyen gioi han, **huong nao dung duoc ha tang da co** thuong la huong toi uu nhat.

### 4. No cho phep rollout theo domain, giup toi uu hoa chi phi GPU

Toi khong muon:

- train depth prior cho ca 7 scene roi moi phat hien 2 scene generic te di.

Toi muon:

- chi test sau tren scene benchmark,
- roi rollout co dieu kien theo domain.

Dieu nay rat hop voi bai toan gioi han tai nguyen:

- it run hon,
- ket luan ro hon,
- tranh brute-force.

## Cai tien nay nen duoc hieu nhu mot "goi nang cap" chinh thuc

Toi khong de xuat "bat DEPTH_PRIOR=1" mot cach ngau hung.

Toi de xuat mot goi cai tien hoan chinh gom 5 thanh phan:

### Thanh phan 1: Baseline certification truoc khi bat depth

Khong co baseline sach, depth prior se bi oan hoac duoc thuong oan.

Bat buoc:

- holdout co dinh;
- antialiasing khoa cung;
- sanity-check render bat buoc;
- score chi chap nhan khi run hop le.

Neu khong, moi ket qua "depth co tang diem" deu khong dang tin.

### Thanh phan 2: Depth prior chi danh vao domain BTS truoc

Thu tu dung:

- benchmark scene BTS kho;
- benchmark scene BTS trung binh;
- 1 scene generic de lam doi chung.

Muc tieu:

- xac nhan loi co tinh he thong o BTS;
- xac nhan generic co nen giu baseline thuong hay khong.

Day la logic cua nguoi thi chuyen nghiep:

- tim noi depth "an diem that" truoc;
- khong ep rollout tren noi no khong mang lai gia tri.

### Thanh phan 3: Rule rollout theo domain

Sau A/B, rollout theo mot trong 3 che do:

- `Policy A`
  - BTS: depth prior on
  - Generic: depth prior off

- `Policy B`
  - BTS: depth prior on
  - Generic: depth prior on

- `Policy C`
  - BTS: depth prior off
  - Generic: depth prior off

Trong thuc te, toi danh gia `Policy A` la kha nang cao nhat neu repo va domain dung nhu mo ta hien tai.

### Thanh phan 4: Chi tiet tai nguyen de depth prior khong thanh "lo dot GPU"

Voi tai nguyen gioi han, toi se dat quy tac:

- khong train full 7 scene de test;
- chi train benchmark scene cho moi policy;
- sau khi policy chot moi retrain final.

Toi se gioi han ngan sach thu nghiem nhu sau:

- 2-3 run baseline benchmark
- 2-3 run depth benchmark
- 1 quyet dinh rollout
- 1 lan retrain final

Luc nay depth prior van la cai tien lon, nhung da bi "dong khung van hanh" de khong an sach ngan sach GPU.

### Thanh phan 5: Booster chi duoc phep dung sau khi domain-depth policy da chot

Neu depth prior cho BTS tang diem that, khi do moi duoc quyen thu:

- TRR refine nhu lop booster.

Vi sao:

- khi geometry da duoc nang cap, moi booster moi co nen tot de hoat dong;
- neu geometry chua on, booster chi lam ket qua kho phan tich hon.

Noi cach khac:

- depth prior la thuoc tri benh;
- TRR la thang toi uu hoa sau dieu tri.

## Mo phong quyet dinh neu toi la nguoi chi huy du an

Neu toi phai dua ra mot quyet dinh ngay hom nay ma khong duoc mo rong vo tan y tuong, toi se ky quyet dinh nhu sau:

### Quyet dinh 1

Dong bang baseline operational:

- sparse BTC
- antialiasing on
- holdout co dinh
- sanity-check render mandatory

### Quyet dinh 2

Mo 1 nhanh chinh thuc:

- `domain-depth-policy`

Nhiem vu duy nhat cua nhanh nay:

- chung minh depth prior co tao ra policy tot hon baseline hay khong.

### Quyet dinh 3

Chi cho phep 3 ket qua hop le:

- giu baseline tat ca
- BTS depth / generic baseline
- tat ca cung depth

Khong cho phep:

- mo rong 5 nhanh y tuong song song
- sua lung tung 4-5 thong so khac nhau cung luc
- chen TRR truoc khi xong geometry A/B

### Quyet dinh 4

Neu `Policy A` thang:

- retrain 5 scene BTS voi depth prior
- giu 2 scene generic theo baseline
- render final
- package

Day la cach toi uu tai nguyen nhat de co score tot nhat ma khong can phat minh them qua nhieu.

## Vi sao day la "cai tien hoan hao nhat" trong boi canh hien tai

Tu "hoan hao" o day khong co nghia la y tuong dep nhat ve hoc thuat.

No co nghia la:

- dung diem dau nhat cua bai toan;
- kha thi voi repo hien tai;
- tan dung ha tang da co;
- co kha nang tang diem that;
- co the van hanh voi tai nguyen gioi han;
- it rui ro hon cac huong sua trainer sau.

Neu xet theo tieu chi doanh nghiep, day la phuong an co:

- ROI cao nhat;
- implementation risk vua phai;
- deployment risk thap;
- kha nang giai thich va truy vet cao.

## Nhung thu toi chu dong khong chon la "cai tien so 1"

### Khong chon TRR lam so 1

Vi:

- no la booster, khong phai geometry core;
- de co artifact;
- phan tich A/B kho hon.

### Khong chon antenna-focus lam so 1

Vi:

- phai patch trainer;
- rui ro tuong thich cao;
- mat nhieu cong debug hon loi ich ky vong ban dau.

### Khong chon hyperparameter sweep rong lam so 1

Vi:

- rat ton GPU;
- de ra ket qua "ngau nhien dep";
- kho truy vet nguyen nhan.

## De xuat chot cuoi cung

Neu can mot cau tra loi duy nhat, day la cau tra loi cua toi:

> Cai tien tot nhat trong dieu kien tai nguyen gioi han la bien depth prior thanh mot chinh sach train theo domain: bat cho 5 scene BTS neu A/B xac nhan loi, giu baseline sach cho 2 scene generic neu depth khong on dinh. Day la huong vua sua geometry that, vua tan dung ha tang san co, vua co kha nang tang diem thuc chat cao nhat tren repo nay.

Neu can mot ten goi chuyen nghiep de dua vao tai lieu van hanh, toi dat ten no la:

> **DADT Policy - Domain-Adaptive Depth Training**

Va neu chi duoc dau tu 1 nhanh nghien cuu tu bay gio den luc chot bai, toi dau tu toan bo vao nhanh nay truoc tat ca cac nhanh con lai.
