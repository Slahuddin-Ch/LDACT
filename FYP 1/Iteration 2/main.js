const {app, BrowserWindow} = require('electron')


function createWindow () {
    window = new BrowserWindow({width: 1100, height: 900})
    window.loadFile('app/index.html')
}

function createWindow1 () {
    /*...*/
    var python = require('child_process').spawn('python', ['./code.py']);
    python.stdout.on('data',function(data){
        console.log("data: ",data.toString('utf8'));
    });
 }


app.on('ready', createWindow)
app.on('ready', createWindow1)


app.on('window-all-closed', () => {
    // On macOS it is common for applications and their menu bar
    // to stay active until the user quits explicitly with Cmd + Q
    if (process.platform !== 'darwin') {
      app.quit()
    }
})
