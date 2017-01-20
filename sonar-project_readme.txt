Steps to Analyze a Python Project
Install SonarQube Server (see Setup and Upgrade for more details)
Install SonarQube Scanner and be sure your can call sonar-scanner from the directory where you have your source code
Install SonarPython (see Installing a Plugin for more details)
(Optional) Install Pylint if you want to activate Pylint rules
Create a sonar-project.properties file at the root of your project (see a sample project on GitHub: https://github.com/SonarSource/sonar-scanning-examples/tree/master/sonarqube-scanner)
Run sonar-scanner command from the project root dir
Follow the link provided at the end of the analysis to browse your project's quality in SonarQube UI.
Advanced Usage
With SonarPython, you can:
import Unit Tests Execution Reports
import Coverage Results
import a Pylint Report
create your own Custom Rules

Get Started in Two Minutes
1. Download and unzip the SonarQube distribution (let's say in "C:\sonarqube" or "/etc/sonarqube")
2. Start the SonarQube server:
# On Windows, execute:
C:\sonarqube\bin\windows-x86-xx\StartSonar.bat
 
# On other operating system, execute:
/etc/sonarqube/bin/[OS]/sonar.sh console
3. Download and unzip the SonarQube Scanner (let's say in "C:\sonar-scanner" or "/etc/sonar-scanner")
4. Download and unzip some project samples (let's say in "C:\sonar-scanning-examples" or "/etc/sonar-scanning-examples")
5. Analyze a project:
# On Windows:
cd C:\sonar-scanning-examples\sonarqube-scanner
C:\sonar-scanner\bin\sonar-scanner.bat
 
# On other operating system:
cd /etc/sonar-scanning-examples/sonarqube-scanner
/etc/sonar-scanner/bin/sonar-scanner
6. Browse the results at http://localhost:9000 (default System administrator credentials are admin/admin)