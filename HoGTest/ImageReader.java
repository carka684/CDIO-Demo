import java.io.File;
import java.util.Vector;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class ImageReader
{
	
	private static Mat classes;
	private static Vector<String> files;
	private static int numberOfPos;
	private static int numberOfNeg;
	
	public ImageReader()
	{
		classes = new Mat();
		files = new Vector<String>();
		numberOfNeg = 0;
		numberOfPos = 0;
	}
	
	public void readImages(String folderPosTrain, String folderNegTrain)
	{
		files.clear();
		classes.release();
		try {

			Vector<String> posFiles = listFilesForFolder(folderPosTrain);
			numberOfPos = posFiles.size();

			Vector<String> negFiles = listFilesForFolder(folderNegTrain);
			numberOfNeg = negFiles.size();
			getFiles().addAll(posFiles);
			getFiles().addAll(negFiles);

			getClasses().push_back(new Mat(numberOfPos,1,CvType.CV_32F, new Scalar(1)));
			getClasses().push_back(new Mat(numberOfNeg,1,CvType.CV_32F, new Scalar(-1)));
		} catch (Exception e) {
			System.out.println("Couldn't load images");
			e.printStackTrace();
		}	
	}
	public static Vector<String> listFilesForFolder(String folder) {
		Vector<String> filesVec = new Vector<String>();
		File[] files = new File(folder).listFiles();
		for (File file : files) {
		    if (file.isFile()) {
		    	filesVec.add(folder+file.getName());
		    }
		}
		return filesVec;
	}
	public Mat getClasses()
	{
		return classes;
	}
	public Vector<String> getFiles()
	{
		return files;
	}
	public int getPos()
	{
		return numberOfPos;
	}
	public int getNeg()
	{
		return numberOfNeg;
	}
}