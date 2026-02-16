@echo off

for %%B in (1 4 16 32) do (
    python src/test.py ^
        --texts-file tests/inputs.txt ^
        --clients-max 1 ^
        --clients-step 1 ^
        --requests-per-client 50 ^
        --batch-size %%B ^
        --expected-dim 1024 ^
        --skip-validation ^
        --output-dir ./results/batch_sizing
)

echo Done
pause