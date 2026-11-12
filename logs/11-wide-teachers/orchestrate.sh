#!/bin/bash
# Waits for both teachers, launches the student, waits for it, then runs the finish stage.
export PATH=$HOME/.local/bin:$PATH
cd ~/resnet-5m
until grep -q "^epoch 250/250" logs/T_wide_s1.log 2>/dev/null && grep -q "^epoch 250/250" logs/T_wide_s2.log 2>/dev/null \
      && grep -q "^saved" <(tail -1 logs/T_wide_s1.log) && grep -q "^saved" <(tail -1 logs/T_wide_s2.log); do sleep 30; done
echo "$(date) teachers finished" >> logs/orchestrate.log
bash .context/round11/run_h200.sh student >> logs/orchestrate.log 2>&1
until grep -q "^epoch 200/200" logs/R11_wide.log 2>/dev/null && grep -q "^saved" <(tail -1 logs/R11_wide.log); do sleep 30; done
echo "$(date) student finished" >> logs/orchestrate.log
bash .context/round11/run_h200.sh finish > logs/finish.log 2>&1
echo "$(date) finish done" >> logs/orchestrate.log
