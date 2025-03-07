# ����ѵ��������ű�
$trainingBatch = @"
@echo off
echo ��ʼִ��ģ��ѵ��... > training_log.txt
echo %date% %time% >> training_log.txt

echo ���� conda ���� academic... >> training_log.txt
call C:\Users\%USERNAME%\Anaconda3\Scripts\activate academic

set PYTHONPATH=%CD%

echo ѵ�� LSTM ģ��... >> training_log.txt
conda run -n academic --no-capture-output python -m prediction.train --model lstm --input_length 20 --output_length 5 --batch_size 64 --lr 0.001 --epochs 100 >> training_log.txt 2>&1

echo ѵ�� Seq2Seq ģ��... >> training_log.txt
conda run -n academic --no-capture-output python -m prediction.train --model seq2seq --input_length 20 --output_length 5 --batch_size 64 --lr 0.001 --epochs 100 >> training_log.txt 2>&1

echo ѵ�� Transformer ģ��... >> training_log.txt
conda run -n academic --no-capture-output python -m prediction.train --model transformer --input_length 20 --output_length 5 --batch_size 64 --lr 0.0005 --epochs 100 >> training_log.txt 2>&1

echo ѵ�� STGCN ģ��... >> training_log.txt
conda run -n academic --no-capture-output python -m prediction.train --model stgcn --input_length 20 --output_length 5 --batch_size 64 --lr 0.001 --epochs 100 >> training_log.txt 2>&1

echo ����ģ��ѵ�����! >> training_log.txt
echo %date% %time% >> training_log.txt
"@

# ���� run_training.bat
Set-Content -Path "run_training.bat" -Value $trainingBatch -Encoding UTF8

# ���������ű�
$startBatch = @"
@echo off
echo ���� Conda ���� academic...
call C:\Users\%USERNAME%\Anaconda3\Scripts\activate academic
echo ִ�� run_training.bat...
call "%~dp0run_training.bat"
"@

# ���� start_training.bat
Set-Content -Path "start_training.bat" -Value $startBatch -Encoding UTF8

# ����2Сʱ���ִ��ʱ��
$triggerTime = (Get-Date).AddHours(5)
Write-Host "������ $triggerTime ִ��"

# ȷ�� start_training.bat ������·��
$startTrainingPath = "$PWD\start_training.bat"

# �����ƻ�����
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$startTrainingPath`""
$trigger = New-ScheduledTaskTrigger -Once -At $triggerTime
$settings = New-ScheduledTaskSettingsSet -DontStopOnIdleEnd -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

# ע��ƻ����񣨿�����Ҫ����ԱȨ�ޣ�
Register-ScheduledTask -TaskName "TrainTrajectoryModels" -Action $action -Trigger $trigger -Settings $settings -Force

Write-Host "������5Сʱ���Զ�ִ��ģ��ѵ������־����¼�� training_log.txt ��"
