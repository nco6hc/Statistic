@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:menu
cls
echo ================================================================
echo        STUDENT SELECTION PREDICTOR  (M2+ Narrow Adaptive)
echo ================================================================
echo.
echo   1. PREDICT    - Generate 5 candidate groups for a day
echo   2. CORRECT    - Submit actual results after class
echo   3. STATUS     - View accuracy history
echo   4. EXIT       - Close program
echo.
echo ================================================================

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto predict
if "%choice%"=="2" goto correct
if "%choice%"=="3" goto status
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:predict
cls
echo ================================================================
echo                    GENERATE PREDICTIONS
echo                  (M2+ Narrow Adaptive, 5 Groups)
echo ================================================================
echo.
set /p day="Enter day number (e.g., 1310): "
echo.
echo Generating predictions for Day %day%...
echo.
python m2_predictor.py predict %day%
echo.
echo ================================================================
pause
goto menu

:correct
cls
echo ================================================================
echo                   SUBMIT CORRECTIONS
echo ================================================================
echo.
set /p day="Enter day number (e.g., 1310): "
echo.
echo Enter the 6 student IDs that were actually selected
echo Format: student1,student2,student3,student4,student5,student6
echo Example: 6,11,22,33,55,12
echo.
set /p students="Enter student IDs: "
echo.
echo Submitting correction for Day %day%...
echo.
python m2_predictor.py correct %day% %students%
echo.
echo ================================================================
pause
goto menu

:status
cls
echo ================================================================
echo                     SYSTEM STATUS
echo ================================================================
echo.
python m2_predictor.py status
echo.
echo ================================================================
pause
goto menu

:exit
cls
echo.
echo Thank you for using the M2+ Narrow Adaptive Predictor!
echo.
timeout /t 2 >nul
exit
