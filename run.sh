num_worker=$1
beta=10.0
S=2
tau=1
steps=50
device=cpu:0

python knownhosts_gen.py $num_worker

python main.py $num_worker --beta $beta --S $S --tau $tau --steps $steps --device $device &

for w_i in $(seq 0 $(expr $num_worker - 1))
do
    python main.py $w_i --beta $beta --S $S --tau $tau --steps $steps --device $device &
done
