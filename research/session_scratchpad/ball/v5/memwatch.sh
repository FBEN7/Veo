#!/bin/sh
# Log used memory every 30 s (to see whether restarts follow memory peaks).
while true; do echo "$(date -u +%H:%M:%S) $(free -m | awk '/Mem:/{print $3, $7}') $(pgrep -c eval_ball)" >> /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5/mem.log; sleep 30; done
