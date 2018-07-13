using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Xml;
using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Structure;

namespace frec
{
    class Program
    {
        static string xmlDir = $@"C:\Users\chris.boseak\Desktop\sdfsdf\training\haarcascades\haarcascade_frontalface_alt_tree.xml";
        static CascadeClassifier Classifier = new CascadeClassifier(xmlDir);

        static EigenFaceRecognizer faceRecognizerEigen = new EigenFaceRecognizer(80, double.PositiveInfinity);
        static LBPHFaceRecognizer faceRecognizerLBPH = new LBPHFaceRecognizer(1, 8, 8, 8, double.PositiveInfinity);
        static FisherFaceRecognizer faceRecognizerFisher = new FisherFaceRecognizer(0, double.PositiveInfinity);
         static void Main(string[] args)
        {


            try
            {

                var elvisPics = getAllImages(ProcessDirectory(@"C:\Users\chris.boseak\Desktop\sdfsdf\training\labelled\elvis"));
                var jacksonPics = getAllImages(ProcessDirectory(@"C:\Users\chris.boseak\Desktop\sdfsdf\training\labelled\jackson"));
                var cowellPics = getAllImages(ProcessDirectory(@"C:\Users\chris.boseak\Desktop\sdfsdf\training\labelled\cowell"));

                var trainingData = new List<Image<Gray, Byte>>();
                var trainingLabels = new List<int>();

                AddToTrainingData(ref trainingData, ref trainingLabels, elvisPics, 1);
                AddToTrainingData(ref trainingData, ref trainingLabels, jacksonPics, 2);
                AddToTrainingData(ref trainingData, ref trainingLabels, cowellPics, 3);


                faceRecognizerEigen.Train<Gray, Byte>(trainingData.ToArray(), trainingLabels.ToArray());
                faceRecognizerLBPH.Train<Gray, Byte>(trainingData.ToArray(), trainingLabels.ToArray());
                faceRecognizerFisher.Train<Gray, Byte>(trainingData.ToArray(), trainingLabels.ToArray());

                var elvisTestImgs =
                    getAllImages(ProcessDirectory(@"C:\Users\chris.boseak\Desktop\sdfsdf\training\labelled\elvis test"));
                var elvisFakeImgs =
                    getAllImages(ProcessDirectory(@"C:\Users\chris.boseak\Desktop\sdfsdf\training\labelled\elvis fake"));
                var cowellTestImgs =
                    getAllImages(ProcessDirectory(@"C:\Users\chris.boseak\Desktop\sdfsdf\training\labelled\cowell test"));



                Console.WriteLine("*****************SHOULD PASS - 1**************************");

                CheckImagesAndPrintResult(elvisTestImgs);

                Console.WriteLine("*****************SHOULD PASS - 3**************************");
                CheckImagesAndPrintResult(cowellTestImgs);



                Console.WriteLine("*****************SHOULD FAIL**************************");
                CheckImagesAndPrintResult(elvisFakeImgs);
               
              

              

            }
            catch (Exception ex)
            {
                throw ex;
            }


        }

        static void CheckImagesAndPrintResult(List<Image<Gray, Byte>> imgs)
        {
            foreach (var i in imgs)
            {

                var EigenRes = faceRecognizerEigen.Predict(i);
                var LBPHRes = faceRecognizerLBPH.Predict(i);
                var FisherRes = faceRecognizerFisher.Predict(i);
                var EigenMatch = (EigenRes.Distance < 5000);
                var LBPHMatch = (LBPHRes.Distance < 25);
                var FisherMatch = (LBPHRes.Distance < 300);

                if (Convert.ToInt32(EigenMatch) + Convert.ToInt32(LBPHMatch) + Convert.ToInt32(FisherMatch) < 2)
                {
                    Console.WriteLine("No Match");
                }

                else
                {
                    if (EigenMatch)
                        Console.WriteLine($"{EigenRes.Label}\t{EigenRes.Distance}");
                    if (LBPHMatch)
                        Console.WriteLine($"{LBPHRes.Label}\t{LBPHRes.Distance}");
                    if (FisherMatch)
                        Console.WriteLine($"{FisherRes.Label} \t{FisherRes.Distance}");
                }

                Console.WriteLine("*******************************************************");
            }

        }

        static void AddToTrainingData(ref List<Image<Gray, Byte>> data, ref List<int> labels, List<Image<Gray, Byte>> inImgs, int label)
        {
            foreach (var e in inImgs)
            {
                data.Add(e);
                labels.Add(label);
            }

        }
        static string[] ProcessDirectory(string targetDirectory)
        {
            // Process the list of files found in the directory.
            return Directory.GetFiles(targetDirectory);

        }

        static List<Image<Gray, Byte>> getAllImages(string[] paths, int minSize = 300)
        {
            var count = 0;

            var Rand = new Random();
            var ret = new List<Image<Gray, Byte>>();
            foreach (var p in paths)
            {

                var img = loadGrayImage(p);//.Resize(minSize, minSize, Emgu.CV.CvEnum.Inter.Cubic);


                var faces = Classifier.DetectMultiScale(img.Mat);
                if (!faces.Any()) continue;

                var rect = faces.First();
                var crop = img.GetSubRect(rect);

                ret.Add(new Image<Gray, Byte>(CropFace(crop, minSize)));





            }
            return ret;
        }

        static Bitmap CropFace(Bitmap bmp, int size)
        {
            var ret = new Bitmap(bmp, new Size(size, size));
         //   ret.Save($@"C:\Users\chris.boseak\Desktop\sdfsdf\training\labelled\crops\{bmp.GetHashCode()}.bmp");
            return ret;
        }
        static Bitmap CropFace(Image<Gray, Byte> img, int size)
        {
            return CropFace(img.ToBitmap(), size);
        }
        static Bitmap MarkFace(Bitmap inFile, Rectangle[] rects)
        {
            foreach (var rect in rects)
            {
                using (Graphics gr = Graphics.FromImage(inFile))
                {
                    gr.SmoothingMode = SmoothingMode.AntiAlias;
                    //  gr.FillEllipse(Brushes.LightGreen, rect);
                    using (Pen thick_pen = new Pen(Color.Red, 5))
                    {
                        gr.DrawRectangle(thick_pen, rect);
                    }
                }
            }


            return inFile;
        }

        static Image<Gray, Byte> loadGrayImage(string path)
        {
            var img1 = new Image<Gray, Byte>(path);
            return img1;
        }
        static Image<Bgr, Byte> loadBgrImage(string path)
        {
            var img1 = new Image<Bgr, Byte>(path);
            return img1;
        }

    }
}
