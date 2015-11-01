mkfifo /tmp/player1 /tmp/player2
tee p1_in < /tmp/player1 | $1 | tee p1_out | nc 127.0.0.1 64335 > /tmp/player1 &
pid1=$!
tee p2_in < /tmp/player2 | $2 | tee p2_out | nc 127.0.0.1 64335 > /tmp/player2 &
pid2=$!
while kill -0 $pid1 $pid2; do sleep 1; done
kill $pid1 $pid2
rm /tmp/player1 /tmp/player2
