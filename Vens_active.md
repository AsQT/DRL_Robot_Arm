Cứ nhớ một câu thôi: venv của bạn nằm ở repo root, tức là:

Nếu terminal đang ở repo root:
```python
.\.venv\Scripts\Activate.ps1
```
Nếu terminal đang ở trong project con DRL_Pathplanning_trainning:
```python
..\.venv\Scripts\Activate.ps1
```

Kiểm tra đã kích hoạt đúng chưa:
```python
python -c "import sys; print(sys.executable)"
```
Kết quả đúng phải là:
```python
C:\Users\MinhQuang\DRL\.venv\Scripts\python.exe
```
```python
C:\Users\MinhQuang\DRL\.venv
```

Sau đó mới chạy script:
```python
python .\DRL_Pathplanning_trainning\Evaluation\test_environment_start_to_target.py --config .\DRL_Pathplanning_trainning\config\environment.yaml --episodes 3 --steps 50 --gui false --show false
```

Nếu PowerShell báo không cho chạy script, chạy tạm lệnh này trong terminal đó:
```python
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
rồi activate lại.

Muốn thoát môi trường:
```python
deactivate
```
