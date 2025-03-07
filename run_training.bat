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
