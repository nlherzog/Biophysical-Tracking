//Select Directories
input = getDirectory("Choose Source Directory");
output = getDirectory("Choose Destination Directory");

//===============================
setBatchMode("hide");							// that does not have the popup window
run("Bio-Formats Macro Extensions"); // need this to run to open images with the special importer (see below)
		
		
list = getFileList(input);			//a new variable containing the names of the files in the input variable
for (i = 0; i < list.length; i++)
        action(input, output, list[i]);
 {
function action(input, output, filename) {
        Ext.openImagePlus(input+list[i]); //this is the line to open images without the bioformats importer
        filename = getTitle();
        //run("Split Channels");
        //selectWindow("C2-" + filename);  // change channel here, use this for multiple channel images)
        selectWindow(filename); //use this line of code for a single channel z-stack
        run("Make Subset...", "slices=1");
        saveAs("Tiff", output + filename);
        run("Close All");}
}
