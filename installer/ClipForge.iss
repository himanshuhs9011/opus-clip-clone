; ClipForge setup wizard (Inno Setup 6). Built by installer\build.ps1:
;   ISCC /DStageDir=<staged app files> /DOutDir=<output folder> ClipForge.iss
#ifndef StageDir
  #define StageDir "build\stage"
#endif
#ifndef OutDir
  #define OutDir "..\dist"
#endif
#define AppName "ClipForge"
#define AppVersion "2.1.0"
#define RepoURL "https://github.com/himanshuhs9011/opus-clip-clone"

[Setup]
AppId={{8F3C2A51-6E7B-4C1D-9A2E-5B7C1D3E4F60}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppName}
AppPublisherURL={#RepoURL}
AppSupportURL={#RepoURL}/issues
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutDir}
OutputBaseFilename=ClipForge-Setup
SetupIconFile={#StageDir}\clipforge.ico
UninstallDisplayIcon={app}\clipforge.ico
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nClipForge turns long videos into short, captioned, vertical clips and ranks them by how viral they are likely to be. Everything runs on your own PC.%n%nSetup downloads the remaining parts by itself (Python, AI, video tools). Keep the internet connected.
FinishedLabel=ClipForge is ready.%n%nYour clips are saved in Videos\ClipForge.

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "ollama"; Description: "Install Ollama, the free local AI that picks clips (about 1 GB)"; GroupDescription: "AI:"; Check: not OllamaInstalled
Name: "model"; Description: "Download the local AI model Gemma 4 (gemma4:e4b, about 6.6 GB, recommended)"; GroupDescription: "AI:"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "install.ps1"; DestDir: "{app}"; Flags: ignoreversion

[InstallDelete]
; leftovers from the first (non-wizard) installer
Type: files; Name: "{userprograms}\ClipForge.lnk"
Type: files; Name: "{userprograms}\Uninstall ClipForge.lnk"
Type: files; Name: "{app}\uninstall.ps1"

[Registry]
; remove the first installer's "Apps" entry so ClipForge is listed once
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\ClipForge"; Flags: deletekey dontcreatekey

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\ClipForge.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\clipforge.ico"; Comment: "Turn long videos into viral shorts"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\ClipForge.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\clipforge.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\ClipForge.pyw"""; WorkingDir: "{app}"; Description: "Open ClipForge"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -Command ""Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*ClipForge.pyw*' } | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }"""; Flags: runhidden; RunOnceId: "StopClipForge"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\bin"
Type: filesandordirs; Name: "{app}\assets"
Type: filesandordirs; Name: "{app}\output"
Type: filesandordirs; Name: "{app}\uploads"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\opusclip\__pycache__"
Type: files; Name: "{app}\.env"
Type: files; Name: "{app}\install.log"
Type: dirifempty; Name: "{app}"

[Code]
var
  KeyPage: TInputQueryWizardPage;

function OllamaInstalled: Boolean;
begin
  Result := FileExists(ExpandConstant('{localappdata}\Programs\Ollama\ollama.exe'));
end;

procedure InitializeWizard;
begin
  KeyPage := CreateInputQueryPage(wpSelectTasks,
    'Cloud AI (optional)',
    'Use OpenRouter as well as, or instead of, the local AI',
    'If you have an OpenRouter API key, paste it below. You can leave it empty and add it later in ClipForge, Settings.');
  KeyPage.Add('OpenRouter API key:', True);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Params, Key: String;
  ShowCmd, ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    exit;
  WizardForm.StatusLabel.Caption :=
    'Downloading and setting up Python, AI and video tools. A window shows the progress; this can take 5 to 20 minutes.';
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\install.ps1') +
            '" -Wizard -AppDir "' + ExpandConstant('{app}') + '"';
  if (not OllamaInstalled) and (not WizardIsTaskSelected('ollama')) then
    Params := Params + ' -NoOllama';
  if not WizardIsTaskSelected('model') then
    Params := Params + ' -NoModel';
  Key := Trim(KeyPage.Values[0]);
  if Key <> '' then
    Params := Params + ' -OpenRouterKey "' + Key + '"';
  ShowCmd := SW_SHOWNORMAL;
  if WizardSilent then
    ShowCmd := SW_HIDE;
  if not Exec('powershell.exe', Params, '', ShowCmd, ewWaitUntilTerminated, ResultCode) then
    MsgBox('Setup could not start the component installer.', mbError, MB_OK)
  else if ResultCode = 1 then
    MsgBox('Some components could not be installed. Check your internet connection and run the setup again.' + #13#10 +
           'Details: ' + ExpandConstant('{app}\install.log'), mbError, MB_OK)
  else if ResultCode = 2 then
    MsgBox('ClipForge is installed. A few optional parts need attention (for example the AI model download).' + #13#10 +
           'Details: ' + ExpandConstant('{app}\install.log'), mbInformation, MB_OK);
end;
