import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.objdetect.HOGDescriptor;

import edu.wildlifesecurity.framework.identification.impl.HOGIdentification;



public class hello
{
	public static void main( String[] args )
	{
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		
		HOGIdentification hogTest = new HOGIdentification(); 
		hogTest.init();
		hogTest.trainClassifier("Images/trainIm/pos/", "Images/trainIm/neg");

		//		testSVM();
		//testHog();
	}

	public static void testHog()
	{
		Vector<String> trainFiles = new Vector<String>();
		ImageReader trainReader = new ImageReader();
		trainReader.readImages("Images/trainIm/pos/", "Images/trainIm/neg/");
		trainFiles = trainReader.getFiles();
		Mat classes = new Mat();
		classes = trainReader.getClasses();
		
		Size s = new Size(480,480);
		
		HOGDescriptor hog = new HOGDescriptor(s,new Size(16,16),new Size(8,8),new Size(8,8),9,-1,0.2,1,1,false,64);
		Mat trainFeat = getDescriptors(trainFiles, hog, s);
		
		System.out.println(trainFeat.size().toString());
		System.out.println(classes.size().toString());
		CvSVM SVM = new CvSVM();
		CvSVMParams params = new CvSVMParams();
	    params.set_kernel_type(CvSVM.LINEAR);
	    long startTime = System.nanoTime();
		SVM.train(trainFeat,classes,new Mat(),new Mat(),params);
		long endTime = System.nanoTime();
		System.out.println((endTime-startTime)/1000000);
		SVM.save("test.txt");
		
		predictSVM(SVM,hog,s);
		
	}
	public static  double[] getResult(Mat classes, Mat result,int numberOfPos, int numberOfNeg)
	{
		
		Mat falseNegMat = new Mat();
		Mat falsePosMat = new Mat();
		Core.absdiff(classes.rowRange(0, numberOfPos),result.rowRange(0, numberOfPos),falseNegMat);
		Core.absdiff(classes.rowRange(numberOfPos,numberOfPos+numberOfNeg),result.rowRange(numberOfPos, numberOfPos+numberOfNeg),falsePosMat);
		
		Scalar falseNegRes =  Core.sumElems(falseNegMat);
		Scalar falsePosRes =  Core.sumElems(falsePosMat);
		double FN = falseNegRes.mul(new Scalar((double) 1/(2*numberOfPos))).val[0];
		double TP = 1-FN;
		double FP = falsePosRes.mul(new Scalar((double) 1/(2*numberOfPos))).val[0];
		double TN  = 1 - FP;
		double[] res = {TP,FN,TN,FP};
		return res;
		
	}
	public static Mat getDescriptors(Vector<String> strVec,HOGDescriptor hog,Size s)
	{
		
		MatOfFloat descriptors = new MatOfFloat();
		Mat m;
		Mat featMat = new Mat();
		for(String file : strVec)
		{
			m=Highgui.imread(file,Highgui.CV_LOAD_IMAGE_GRAYSCALE);
			Imgproc.resize(m, m, s);
			long startTime = System.nanoTime();
			hog.compute(m, descriptors);			
			long endTime = System.nanoTime();
			System.out.println((endTime-startTime)/1000000);
			featMat.push_back(descriptors.t());
		}
		return featMat;
	}
	public static void predictSVM(CvSVM SVM,HOGDescriptor hog, Size s)
	{
		Vector<String> valFiles = new Vector<String>();
		ImageReader valReader = new ImageReader();
		valReader.readImages("Images/valIm/pos/", "Images/valIm/neg/");
		valFiles = valReader.getFiles();
		Mat classes = new Mat();
		classes = valReader.getClasses();
		
		Mat results = new Mat();
		Mat valFeat = getDescriptors(valFiles, hog, s);
		SVM.predict_all(valFeat, results);
		System.out.println("Classes: " + classes.dump());
		System.out.println("Result" + results.dump());
		System.out.println(valFiles);
		double[] res = getResult(classes, results, valReader.getPos(), valReader.getNeg());
		System.out.println("TP: " + res[0] + " FN: " + res[1]  + " TN: " + res[2] + " FP: " + res[3]);

	}
	public static void testSVM()
	{
		double[][] bufferTrain = {{3,2},{1,2},{0,1},{2,0},{10,11},{8,11},{10,8},{9,8}};
		double[] bufferClasses = {1,1,1,1,-1,-1,-1,-1};

		Mat samples = new Mat(bufferTrain.length,2,CvType.CV_32F);
		for(int i = 0; i < 8; i++)
		{
			samples.put(i, 0, bufferTrain[i]);
		}
		System.out.println(samples.dump());

		Mat classes = new Mat(bufferClasses.length,1,CvType.CV_32F);
		classes.put(0, 0, bufferClasses);
		System.out.println(classes.dump());
		
		CvSVM SVM = new CvSVM();
		CvSVMParams params = new CvSVMParams();
	    params.set_kernel_type(CvSVM.LINEAR);
		SVM.train(samples,classes,new Mat(),new Mat(),params);
		Mat val = new Mat(1,2,CvType.CV_32F);
		double[] value = {13,14};
		val.put(0, 0, value);
		System.out.println(val.dump());
		System.out.println("Result: " + SVM.predict(val));
		SVM.save("test.txt");		
	}
}