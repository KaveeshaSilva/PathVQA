imgv=1
baseUrl=drive/MyDrive/PathVQA
dir=scratch_imgv${imgv}
for seed in 1
do



  # Save logs and models under snap/vqa; make backup.
  # See Readme.md for option details.
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:./src \
      python ${baseUrl}/baselines/method1/src/compareModels.py \
done
