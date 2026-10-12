; Inno Setup script. Build with:
;   ISCC /DAppVersion=0.1.0 /DSourceDir=dist\TTSLTS /DIconFile=icons\ttslts.ico /DOutputDir=installer packaging\windows\ttslts.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\TTSLTS"
#endif
#ifndef IconFile
  #define IconFile "..\..\icons\ttslts.ico"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\installer"
#endif

[Setup]
AppId={{6F1A6C7E-0E7B-4B52-9C1D-7A4F5D2E9B31}
AppName=TTSLTS
AppVersion={#AppVersion}
AppPublisher=TTSLTS
DefaultDirName={autopf}\TTSLTS
DefaultGroupName=TTSLTS
UninstallDisplayIcon={app}\TTSLTS.exe
SetupIconFile={#IconFile}
OutputDir={#OutputDir}
OutputBaseFilename=TTSLTS-Setup
Compression=lzma2
SolidCompression=yes
; Installs for the current user by default, no administrator prompt.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\TTSLTS"; Filename: "{app}\TTSLTS.exe"
Name: "{group}\Uninstall TTSLTS"; Filename: "{uninstallexe}"
Name: "{autodesktop}\TTSLTS"; Filename: "{app}\TTSLTS.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\TTSLTS.exe"; Description: "Launch TTSLTS"; Flags: nowait postinstall skipifsilent
