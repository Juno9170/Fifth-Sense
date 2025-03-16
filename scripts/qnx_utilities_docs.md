# QNX Tips/Tricks  

! We are not QNX, which has its own [GitLab](https://gitlab.com/qnx) and [documentation](https://www.qnx.com/download/group.html?programid=29183) !  

This file provides advice and insights on using the utilities included in `qnx_utilities.zip`,  
available for free at the following Google Drive link:  

[Download qnx_utilities.zip](https://drive.google.com/file/d/1RkwtFA4nmSMfLx1-efZJLXhvE1XsJtW9/view)  

> **Note:** All scripts are designed for UNIX-based systems. If you're using Windows, run them in Git Bash to ensure compatibility.  

## General QNX Utilities Startup Tips & Tricks  

- **Default user passwords** are the same as the username.  
- To enable SSH access on your Raspberry Pi, you may need to create the `/boot/wpa_supplicant.conf` file if it does not exist.  
  Then, paste the following content and replace the placeholder values:  

  ```bash
  network={
      ssid="[network-name]"
      key_mgmt=WPA-PSK
      psk="[network-password]"
  }

* Finding your Raspberry Pi's IP address:
    - The IP address is displayed on the main screen.
    - If you have a terminal open, press Alt+Tab to switch back to the main screen.
# QNX Utilities Setup
- SSH access is required for all utilities. Ensure you can connect to your Raspberry Pi over SSH before using them.
- QNX does not include make by default, which is required for development using QNX SDP.
    - Run make.sh to install make automatically on your Raspberry Pi.
- QNX lacks a package manager, so you cannot install packages conventionally.
    - Use transfer-files.sh to copy pre-compiled binaries to your Raspberry Pi.
# QNX Utilities Usage
`make.sh`
- Installs make on your Raspberry Pi.
- Requires SSH access.
- **Usage:**
    ```bash
    ./make.sh

```transfer-files.sh```
- Copies files or folders from your local system to the Raspberry Pi.
- Prompts for a source and destination:
    - If the source is a file, it is transferred as-is.
    - If the source is a folder, only non-script files inside it are transferred.
- **Usage:**
    ```bash
    ./transfer-files.sh
    
    Enter Source Path: [source]
    Enter Destination Path: [destination]