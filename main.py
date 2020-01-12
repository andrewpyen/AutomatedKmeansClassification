from functiondefs import filterImages, downloadImages, compositeImages, classify

print("""
Welcome to the Unsupervised Classifier. To get started, provide inputs for the desired imagery. 
This program uses Landsat 8 images.

for more information visit https://earthexplorer.usgs.gov/ \n""")


user_save_path = input("Choose a directory to save the Landsat Images: (This directory should not contain any "
                       "other Landsat 8 data!!) \n")

print("Set criteria for filtering the images: ")
downloadImages(str(user_save_path))
print("Starting the download...")

print("Scene files downloaded to ", str(user_save_path))

print("Compositing images...")
compositeImages(user_save_path)
print("The composite has been saved to each scene folder in " +
      str(user_save_path) +
      "/composite_folder as 'composite_bands'.")


clusters = int(input("Choose a number of clusters: \n"))
print("Keep in mind that a higher number of clusters will result in longer processing times. \n"
      "Additionally, a large number of clusters may render the image unreadable for qualitative analysis. \n"
      "It's best to set the number of clusters to a number between 6 and 12. \n")
print("You will be able to select up to three bands to analyze. \n")

band1 = input("Choose first band: \n")
band2 = input("Choose second band: \n")
band3 = input("Choose third band: \n")
print("Classifying Images...")

for path in compositeImages(user_save_path):
    classify(path, clusters, band1, band2, band3)

