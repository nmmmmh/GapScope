# move.sh Q a b Q2 gpu conc : move lines a..b of queue Q (not yet launched) to new queue Q2 and start a worker on it
Q=$1; a=$2; b=$3; Q2=$4; g=$5; n=$6; cd "$(dirname "$Q")"
pos=$(cat $Q.pos); [ $pos -ge $((a-1)) ] && { echo "too late pos=$pos"; exit 1; }
sed -n "${a},${b}p" $Q > $Q2
sed -i "${a},${b}s/.*//" $Q
nohup bash worker.sh $Q2 $g $n > /dev/null 2>&1 &
echo moved $(grep -c . $Q2) to $Q2 gpu $g; grep -c . $Q
