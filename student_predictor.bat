@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:menu
cls
echo ================================================================
echo           STUDENT SELECTION PREDICTION SYSTEM
echo ================================================================
echo.
echo   1. PREDICT    - Generate predictions for a day
echo   2. CORRECT    - Submit actual results after class
echo   3. TRAIN      - Update model with corrections
echo   4. STATUS     - View current status
echo   5. EXIT       - Close program
echo.
echo ================================================================

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto predict
if "%choice%"=="2" goto correct
if "%choice%"=="3" goto train
if "%choice%"=="4" goto status
if "%choice%"=="5" goto exit
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:predict
cls
echo ================================================================
echo                    GENERATE PREDICTIONS
echo ================================================================
echo.
set /p day="Enter day number (e.g., 1310): "
echo.
echo Generating predictions for Day %day%...
echo.
python prediction_api.py predict %day%
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
python prediction_api.py correct %day% %students%
echo.
echo ================================================================
pause
goto menu

:train
cls
echo ================================================================
echo                    UPDATE MODEL
echo ================================================================
echo.
echo This will update the model with all collected corrections.
echo This should be done every 1-2 weeks.
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Training cancelled.
    timeout /t 2 >nul
    goto menu
)
echo.
echo Updating model...
echo.
python prediction_api.py update
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
python prediction_api.py status
echo.
echo ================================================================
pause
goto menu

:exit
cls
echo.
echo Thank you for using Student Selection Prediction System!
echo.
timeout /t 2 >nul
exit
