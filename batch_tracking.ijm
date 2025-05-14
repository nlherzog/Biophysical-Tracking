//Select Directories
dir1 = getDirectory("Choose Source Directory ");
dir2 = getDirectory("Choose Destination Directory Results ");
//===============================

setBatchMode("hide");
run("Bio-Formats Macro Extensions");

//Looping function through the files in a directory

list = getFileList(dir1);
for (i=0; i<list.length; i++) 
{
    if (endsWith(list[i], ".tif"))      // change here the file extension if you have other than .tif
    { 
    	Ext.openImagePlus(dir1+list[i]);
    	run("Particle Tracker 2D/3D", "radius=3 cutoff=0.001 per/abs=0.5 link=1 displacement=6 dynamics=Brownian");
    	run("Close All");
    }
}

// Removing the ".tif" from the file name

for (i=0; i<list.length; i++) 
{
    if (endsWith(list[i], ".tif")) 
    {
    	
		origFile = dir1 + "Traj_" + list[i] + ".csv";
		newFile = dir2 + "Traj_" + replace(list[i], ".tif", ".csv");
		while(!File.exists(origFile))
		{
			wait(1000);
		}
		wait(1000);
		s=File.rename(origFile, newFile);
		
		origFile = dir1 + "Traj_" + list[i] + ".txt";
		newFile = dir2 + "Traj_" + replace(list[i], ".tif", ".txt");
		
    	
    }
}
