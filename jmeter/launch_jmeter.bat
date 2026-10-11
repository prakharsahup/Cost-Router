@echo off
REM ============================================================
REM  CostRouter - JMeter GUI Launcher
REM  Opens JMeter GUI with the CostRouter test plan pre-loaded.
REM  Ensure the API server is running before starting a test:
REM     python -m costrouter.api_server 8000
REM ============================================================

REM Uses the JMETER_HOME environment variable if set.
IF "%JMETER_HOME%"=="" SET JMETER_HOME=C:\apache-jmeter-5.6.3
SET TEST_PLAN=%~dp0costrouter_gui_testplan.jmx

IF NOT EXIST "%JMETER_HOME%\bin\jmeter.bat" (
    echo ERROR: JMeter not found at %JMETER_HOME%
    echo Set the JMETER_HOME environment variable to your JMeter install.
    pause
    exit /b 1
)

echo Starting JMeter GUI with CostRouter test plan...
echo API Server should be running at http://127.0.0.1:8000
echo.

"%JMETER_HOME%\bin\jmeter.bat" -t "%TEST_PLAN%"
