//Select Directories
input = getDirectory("Choose Source Directory");
output = getDirectory("Choose Destination Directory");
masks = getDirectory("Choose Masks Directory");

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
        run("Split Channels");
        selectWindow("C1-" + filename);  // change here dependent on which channel you want to do the max projection (C1- - is channel one etc.)
        saveAs("Tiff", masks + "DNA_" + filename);
        run("Z Project...", "projection=[Max Intensity] all");
        saveAs("Tiff", output + filename);
        selectWindow("C4-" + filename);
        saveAs("Tiff", output + "ICP8_" + filename);
        selectWindow("C6-" + filename);
        saveAs("Tiff", output + "5EU_" + filename);
        run("Close All");}
}