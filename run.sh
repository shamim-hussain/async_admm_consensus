num_worker=$1
beta=${2:-1.0}
S=${3:-$1}
tau=${4:-1}
steps=${5:-50}
device=${6:-cpu:0}


echo ----------------------------------

echo num_worker = $num_worker
echo beta = $beta
echo S = $S 
echo tau = $tau
echo steps = $steps
echo device = $device

echo ----------------------------------

echo Generating knownhosts.json
python knownhosts_gen.py $num_worker

echo Starting MASTER
python main.py $num_worker --beta $beta --S $S --tau $tau --steps $steps --device $device &

for w_i in $(seq 0 $(expr $num_worker - 1))
do
    echo Starting WORKER:$w_i
    python main.py $w_i --beta $beta --S $S --tau $tau --steps $steps --device $device &
done

jobs

