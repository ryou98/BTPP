python dsgd_static.py --topo ring -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python dsgt_static.py --topo ring -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python dsgd_dynamic.py --topo OnePeerExp -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python dsgt_dynamic.py --topo ODEquiDyn -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python dsgd_dynamic.py --topo base_k -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python dsgd_ceca.py --topo ceca-2p -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python relaysgd.py --topo relay_binarytree -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python pushpull.py --topo pp -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python dsgd_d2.py --topo ring -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 100000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python dsgd_static.py --topo exponential -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python dsgd_static.py --topo grid -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000

python dsgd_static.py --topo fully_connected -na 8 --gpu_rank 1 -b 16 -lr 0.01 -ms 8000 11000 --model resnet18 --datasets cifar10 --iterations 13000 --record 300 --keep 1  --warm_up 6000