python dsgd_static.py --topo ring -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python dsgt_static.py --topo ring -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python dsgd_dynamic.py --topo OnePeerExp -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python dsgt_dynamic.py --topo ODEquiDyn -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python dsgd_dynamic.py --topo base_k -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python dsgd_ceca.py --topo ceca-2p -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python relaysgd.py --topo relay_binarytree -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python pushpull.py --topo pp -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python dsgd_d2.py --topo ring -na 24 -b 8 -lr 0.01 -ms 100000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python dsgd_static.py --topo exponential -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python dsgd_static.py --topo grid -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000

python dsgd_static.py --topo fully_connected -na 24 -b 8 -lr 0.01 -ms 8000 11000 --model cnn --datasets mnist --iterations 13000 --record 300 --keep 3 -ht --warm_up 6000