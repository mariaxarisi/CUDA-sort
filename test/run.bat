@echo off

del .\test\GTX750.txt
type nul > .\test\GTX750.txt

make clean
make

for /L %%n in (16,1,27) do (
    echo N = %%n >> .\test\GTX750.txt
    make run %%n >> .\test\GTX750.txt
    echo. >> .\test\GTX750.txt
)