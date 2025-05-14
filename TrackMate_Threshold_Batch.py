import sys
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter

from ij import IJ, WindowManager
from java.io import File

from fiji.plugin.trackmate import Model, Settings, TrackMate, SelectionModel, Logger
from fiji.plugin.trackmate.detection import ThresholdDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.gui.displaysettings.DisplaySettings import TrackMateObject
from fiji.plugin.trackmate.features.track import TrackIndexAnalyzer
from fiji.plugin.trackmate.visualization.hyperstack import HyperStackDisplayer
from fiji.plugin.trackmate.visualization.table import TrackTableView
from ij.gui import GenericDialog
from ij.plugin.filter import ThresholdToSelection
from ij.process import ImageProcessor
from ij import IJ
import time

# Ensure UTF-8 encoding for Fiji Jython
reload(sys)
sys.setdefaultencoding('utf-8')

# Main function for image analysis
def Main_action(input_path, output_path, filename):
    print("Processing:", filename)
    
    # Open the image (This assumes image is a .tif, Fiji can open standard image formats such as TIFF)
    imp = IJ.openImage(input_path + filename)
    if imp is None:
        print("Error: Unable to open", filename)
        return
        
    # Set to last frame
    stack_size = imp.getStackSize()
    imp.setSlice(stack_size)
    imp.show()
    
	# Show Threshold Adjuster
    IJ.setAutoThreshold(imp, "Default dark stack no-reset")
    IJ.run("Threshold...")
    
    # Wait for user to adjust threshold and hit "Set"
    time.sleep(15)  # This is currently 20s, adjust the sleep time if needed
    
    # Retrieve the current threshold value after user has set it
    ip = imp.getProcessor()
    min_threshold = ip.getMinThreshold()  # This should now reflect the threshold set by the user
    
    # Prompt user to confirm or modify the threshold
    gd = GenericDialog("Confirm Intensity Threshold") #check this number is the same as the number at the lower set point
    gd.addNumericField("Intensity Threshold:", min_threshold, 1)
    gd.showDialog()
    
    if gd.wasCanceled():
        print("Processing canceled by user for", filename)
        imp.close()
        return
    
    INTENSITY_THRESHOLD = gd.getNextNumber()
    
    # Apply confirmed threshold
    ip.setThreshold(INTENSITY_THRESHOLD, 65535, ImageProcessor.NO_LUT_UPDATE)

    # ----------------------------
    # Create the model object now
    # ----------------------------
    model = Model()
    model.setLogger(Logger.IJ_LOGGER)

    # ------------------------
    # Prepare settings object
    # ------------------------
    settings = Settings(imp)

    # Configure detector - We use the Strings for the keys
    settings.detectorFactory = ThresholdDetectorFactory()
    settings.detectorSettings = {
        'TARGET_CHANNEL': 1,
        'SIMPLIFY_CONTOURS': False,
        'INTENSITY_THRESHOLD': INTENSITY_THRESHOLD,
    }

    # Add filter for spot area
    filter1 = FeatureFilter('AREA', 8.0, False) #this is currently set at ≤8, probably could be set lower 
    settings.addSpotFilter(filter1)


    # Configure tracker
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
    settings.trackerSettings['LINKING_MAX_DISTANCE'] = 2.0
    settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 2.0
    settings.trackerSettings['MAX_FRAME_GAP'] = 1
    settings.trackerSettings['ALLOW_TRACK_MERGING'] = True #can comment this out if using SIMPLE LAP tracker
    settings.trackerSettings['MERGING_MAX_DISTANCE'] = 1.0 #can comment this out if using SIMPLE LAP tracker

    # Add filter for track length
    filter3 = FeatureFilter('TRACK_DURATION', 5, True) #this is currently set at ≥5 frames when frames are counted as frame and not as time
    settings.addTrackFilter(filter3)

    # Add all analyzers
    settings.addAllAnalyzers()

    # -------------------
    # Instantiate TrackMate
    # -------------------
    trackmate = TrackMate(model, settings)

    # --------
    # Process
    # --------
    if not trackmate.checkInput():
        print("Error:", trackmate.getErrorMessage())
        return

    if not trackmate.process():
        print("Error:", trackmate.getErrorMessage())
        return

    # ----------------
    # Display results
    # ----------------
    selectionModel = SelectionModel(model)

    # Read the default display settings
    ds = DisplaySettingsIO.readUserDefault()
    ds.setTrackColorBy(TrackMateObject.TRACKS, TrackIndexAnalyzer.TRACK_INDEX)
    ds.setSpotColorBy(TrackMateObject.TRACKS, TrackIndexAnalyzer.TRACK_INDEX)

    displayer = HyperStackDisplayer(model, selectionModel, imp, ds)
    displayer.render()
    displayer.refresh()

    # Log results
    model.getLogger().log(str(model))

    # ----------------
    # Save results
    # ----------------
    try:
        # Spot table
        spot_table = TrackTableView.createSpotTable(model, ds)
        spot_table_csv_file = File(output_path + filename.replace('.tif', '_spots.csv'))
        spot_table.exportToCsv(spot_table_csv_file)

        # Edge table
        edge_table = TrackTableView.createEdgeTable(model, ds)
        edge_table_csv_file = File(output_path + filename.replace('.tif', '_edges.csv'))
        edge_table.exportToCsv(edge_table_csv_file)

        # Track table
        track_table = TrackTableView.createTrackTable(model, ds)
        track_table_csv_file = File(output_path + filename.replace('.tif', '_tracks.csv'))
        track_table.exportToCsv(track_table_csv_file)

    except Exception as e:
        print("Error saving CSVs:", str(e))
        
    # ---- CLOSE IMAGE IF LOTS OF FILES TO GO THROUGH ----
    #imp.close()

# Select Directories
input_dir = IJ.getDirectory("Choose Source Directory")
output_dir = IJ.getDirectory("Choose Destination Directory")

# Close all open images and clear results
IJ.run("Close All")
IJ.run("Clear Results")

# Get the list of files in the input directory
input_folder = File(input_dir)
file_list = input_folder.list()

# Process each .tif file
for filename in file_list:
    if filename.endswith(".tif"):
        Main_action(input_dir, output_dir, filename)
       