# 创建训练批处理脚本
$trainingBatch = @"
@echo off
echo 开始执行模型训练... > training_log.txt
echo %date% %time% >> training_log.txt

echo 激活 conda 环境 academic... >> training_log.txt
call C:\Users\%USERNAME%\Anaconda3\Scripts\activate academic

set PYTHONPATH=%CD%

echo 训练 LSTM 模型... >> training_log.txt
conda run -n academic --no-capture-output python -m prediction.train --model lstm --input_length 20 --output_length 5 --batch_size 64 --lr 0.001 --epochs 100 >> training_log.txt 2>&1

echo 训练 Seq2Seq 模型... >> training_log.txt
conda run -n academic --no-capture-output python -m prediction.train --model seq2seq --input_length 20 --output_length 5 --batch_size 64 --lr 0.001 --epochs 100 >> training_log.txt 2>&1

echo 训练 Transformer 模型... >> training_log.txt
conda run -n academic --no-capture-output python -m prediction.train --model transformer --input_length 20 --output_length 5 --batch_size 64 --lr 0.0005 --epochs 100 >> training_log.txt 2>&1

echo 训练 STGCN 模型... >> training_log.txt
conda run -n academic --no-capture-output python -m prediction.train --model stgcn --input_length 20 --output_length 5 --batch_size 64 --lr 0.001 --epochs 100 >> training_log.txt 2>&1

echo 所有模型训练完成! >> training_log.txt
echo %date% %time% >> training_log.txt
"@

# 保存 run_training.bat
Set-Content -Path "run_training.bat" -Value $trainingBatch -Encoding UTF8

# 创建启动脚本
$startBatch = @"
@echo off
echo 启动 Conda 环境 academic...
call C:\Users\%USERNAME%\Anaconda3\Scripts\activate academic
echo 执行 run_training.bat...
call "%~dp0run_training.bat"
"@

# 保存 start_training.bat
Set-Content -Path "start_training.bat" -Value $startBatch -Encoding UTF8

# 计算2小时后的执行时间
$triggerTime = (Get-Date).AddHours(5)
Write-Host "任务将在 $triggerTime 执行"

# 确定 start_training.bat 的完整路径
$startTrainingPath = "$PWD\start_training.bat"

# 创建计划任务
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$startTrainingPath`""
$trigger = New-ScheduledTaskTrigger -Once -At $triggerTime
$settings = New-ScheduledTaskSettingsSet -DontStopOnIdleEnd -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

# 注册计划任务（可能需要管理员权限）
Register-ScheduledTask -TaskName "TrainTrajectoryModels" -Action $action -Trigger $trigger -Settings $settings -Force

Write-Host "已设置5小时后自动执行模型训练，日志将记录在 training_log.txt 中"
