Official evaluator test

    Run 2AFC:
        python test_dataset_model.py --dataset_mode 2afc  --datasets val/traditional val/cnn val/color --model lpips --net vgg --use_gpu --batch_size 50


    Run JND:
        python test_dataset_model.py --dataset_mode jnd --datasets val/traditional val/cnn --model lpips --net vgg --use_gpu --batch_size 50

6-candidate evaluation:
    python eval_bapps_6_vgg_candidates.py --root dataset --device cuda --batch_size 64 --num_workers 4 --output bapps_6_vgg_results.csv

Pihlgren's VGG models evaluation:
    python eval_bapps_vgg_arch_layers.py --root dataset --device cuda --batch_size 64 --num_workers 4 --output bapps_vgg_arch_layers_results.csv

Optimization loss evaluation:
    python optimize_image_metric.py --target E:\henrique\PerceptualSimilarity/target.png --input E:\henrique\PerceptualSimilarity/input.png --metrics vgg_r12 vgg_r43 lpips_vgg mse l1 --starts input noisy_target random --resize 256 --steps 1000 --lr 0.03 --save_every 100 --device cuda
