# GNN_mixing_index

Một mô tả ngắn: dự án "GNN_mixing_index" nghiên cứu/triển khai các phương pháp Graph Neural Network (GNN) để tính và phân tích "mixing index" trên đồ thị. README này cung cấp hướng dẫn cài đặt, chạy thử nghiệm, huấn luyện và đánh giá mô hình.

## Mục lục
- [Tổng quan](#tổng-quan)
- [Tính năng chính](#tính-năng-chính)
- [Yêu cầu](#yêu-cầu)
- [Cài đặt](#cài-đặt)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Chuẩn hóa dữ liệu / Dataset](#chuẩn-hóa-dữ-liệu--dataset)
- [Huấn luyện](#huấn-luyện)
- [Đánh giá](#đánh-giá)
- [Kết quả mẫu](#kết-quả-mẫu)
- [Đóng góp](#đóng-góp)
- [Trích dẫn](#trích-dẫn)
- [Giấy phép](#giấy-phép)
- [Liên hệ](#liên-hệ)

## Tổng quan
Giải thích ngắn gọn mục tiêu của repo:
- Mục tiêu: triển khai GNN để tính/ước lượng mixing index và kiểm tra hiệu quả của các kiến trúc GNN khác nhau trên các tập đồ thị (synthetic / thực tế).
- Ứng dụng: phân tích tính đồng nhất/pha trộn lớp, community mixing, hoặc các bài toán liên quan đến cấu trúc đồ thị.


## Tính năng chính
- Triển khai một hoặc nhiều kiến trúc GNN (GCN, GAT, GraphSAGE, ...).
- Tính toán và đánh giá các chỉ số mixing index trên đồ thị.
- Hệ thống training/evaluation có cấu hình (config files).
- Hỗ trợ các tập dữ liệu mẫu và pipeline tiền xử lý.

## Yêu cầu
- Python 3.8+
- PyTorch >= 1.8
- PyTorch Geometric compatible with your PyTorch version
- numpy, pandas, scikit-learn, tqdm, yaml
- (Tuỳ chọn) cuda cho GPU acceleration

Ví dụ file `requirements.txt`:
```
torch>=1.8
torch-geometric
numpy
pandas
scikit-learn
tqdm
pyyaml
```

## Cài đặt
1. Tạo môi trường (conda):
```bash
conda create -n gnn-mix python=3.8 -y
conda activate gnn-mix
```

2. Cài dependencies (pip/conda):
```bash
pip install -r requirements.txt
# hoặc cài torch và torch-geometric theo hướng dẫn chính thức phù hợp với hệ thống CUDA của bạn
```

3. (Tuỳ chọn) Nếu repo có script cài đặt hoặc submodule, chạy:
```bash
# ví dụ
python setup.py install
```

## Cấu trúc dự án (ví dụ)
- data/                # tập dữ liệu raw và đã tiền xử lý
- src/                 # mã nguồn chính (mô hình, dataset, training, evaluation)
- configs/             # cấu hình cho các thí nghiệm (yaml/json)
- experiments/         # checkpoints, logs, kết quả
- notebooks/           # notebook phân tích/visualize
- README.md



## Chuẩn hóa dữ liệu / Dataset
- Mô tả các tập dữ liệu được hỗ trợ (ví dụ: synthetic graphs, Cora, Citeseer, custom graphs).
- Cách chuẩn bị dữ liệu:
  1. Đặt file dữ liệu vào `data/your_dataset/`
  2. Chạy script tiền xử lý:
     ```bash
     python src/preprocess.py --input data/your_dataset/raw --output data/your_dataset/processed
     ```
  3. Cấu hình đường dẫn trong `configs/your_config.yaml`

## Huấn luyện
Ví dụ lệnh huấn luyện chung :
```bash
python src/train.py --config configs/train_config.yaml --device cuda:0
```
Các tham số chính:
- --config : file cấu hình chứa hyperparameters, dataset, output path
- --device : cpu hoặc cuda

Ví dụ cấu hình (configs/train_config.yaml):
```yaml
dataset: your_dataset
model: GCN
epochs: 200
batch_size: 32
learning_rate: 0.001
save_dir: experiments/run1
```

## Đánh giá
Đánh giá mô hình đã huấn luyện:
```bash
python src/eval.py --checkpoint experiments/run1/checkpoint.pth --dataset data/your_dataset/processed
```
Hoặc lệnh để tính mixing index từ graph và output báo cáo:
```bash
python src/compute_mixing_index.py --input data/your_dataset/processed/graph.pt --output results/mixing_index.csv
```

## Kết quả mẫu
(Ghi lại các kết quả mẫu, bảng số liệu, hoặc biểu đồ, ví dụ)
- Model: GCN
- Dataset: synthetic_1
- Accuracy: 0.87
- Mixing index (mean): 0.42

(Cập nhật bằng kết quả thật sau khi chạy experiment)

## Đóng góp
Rất hoan nghênh PR và issue!
- Mở issue mô tả lỗi hoặc yêu cầu tính năng.
- Tạo nhánh (branch) riêng cho tính năng/fix: `git checkout -b feat/your-feature`
- Tạo PR và mô tả thay đổi rõ ràng.

## Trích dẫn
Nếu bạn dùng hoặc trích dẫn công trình này, vui lòng trích dẫn:
```
@misc{yourrepo2026,
  title = {GNN_mixing_index},
  author = {Your Name},
  year = {2026},
  howpublished = {\url{https://github.com/huukhai249/GNN_mixing_index}}
}
```

## Liên hệ
- Tác giả: huukhai249
- Email: huukhai@gnu.ac.kr
- GitHub: https://github.com/huukhai249/GNN_mixing_index

---

Whoever doubts me is gay :)
