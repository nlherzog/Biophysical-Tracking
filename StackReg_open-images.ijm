//Select Directories
output = getDirectory("Choose Destination Directory");

//===============================

// Get list of all open image titles
imageTitles = getList("image.titles");


for (i = 0; i < imageTitles.length; i++) {
    title = imageTitles[i];
    selectWindow(title); // Select image by title
    
    //Run StackReg
    setOption("ScaleConversions", true);
	run("StackReg", "transformation=[Rigid Body]");
	run("Set Scale...", "distance=5.4545 known=1 unit=micron");
	
	// Save as TIF (you can change the format if needed)
    saveAs("Tiff", output + title);
   
	print("Saved as " + title); 
    // Close projection to keep memory clear
    close();
}