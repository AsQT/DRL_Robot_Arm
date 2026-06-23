# PyBullet Control Checks

Chay trong VS Code Terminal tren Windows, profile PowerShell:

```powershell
cd C:\Users\MinhQuang\DRL
.\.venv\Scripts\Activate.ps1
python .\DRL_Robot_Manipulator\Evaluation\PyBullet\Control\test_configuration_space_rand.py
python .\DRL_Robot_Manipulator\Evaluation\PyBullet\Control\test_configuration_space_arm.py
```

Chay nhanh khong mo GUI:

```powershell
python .\DRL_Robot_Manipulator\Evaluation\PyBullet\Control\test_configuration_space_arm.py --headless --samples 10 --sleep 0
python .\DRL_Robot_Manipulator\Evaluation\PyBullet\Control\test_configuration_space_vertices_arm.py --headless --sleep 0
```
