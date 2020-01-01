using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

using System.Drawing;
using System.IO;
using Newtonsoft.Json;


using Emgu.CV;
using Emgu.CV.Structure;

using OpenCvSharp;
using OpenCvSharp.Extensions;

using Accord;
using Accord.Imaging.Filters;
using Accord.Vision.Detection;
using Accord.Vision.Detection.Cascades;


namespace AspNetFrameworkMVC_FaceDetection.Controllers
{

    class Location
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }
    }

    class EmguFaceDetector
    {
        public static List<Rectangle> DetectFaces(Emgu.CV.Mat image)
        {
            List<Rectangle> faces = new List<Rectangle>();
            var facesCascade = HttpContext.Current.Server.MapPath("~/face.xml");
            using ( Emgu.CV.CascadeClassifier face = new Emgu.CV.CascadeClassifier(facesCascade))
            {
                using (UMat ugray = new UMat())
                {
                    CvInvoke.CvtColor(image, ugray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
                    CvInvoke.EqualizeHist(ugray, ugray);
                    Rectangle[] facesDetected = face.DetectMultiScale(
                       ugray,
                       1.1,
                       10,
                       new System.Drawing.Size(20, 20));
                    faces.AddRange(facesDetected);
                }
            }
            return faces;
        }
    }


    class CvSharpFaceDetector
    {
        public static List<OpenCvSharp.Rect> DetectFaces(OpenCvSharp.Mat image)
        {
            List<OpenCvSharp.Rect> faces = new List<OpenCvSharp.Rect>();
            var facesCascade = HttpContext.Current.Server.MapPath("~/face.xml");
            using (OpenCvSharp.CascadeClassifier face = new OpenCvSharp.CascadeClassifier(facesCascade))
            {
                using (OpenCvSharp.Mat ugray = new OpenCvSharp.Mat())
                {
                    Cv2.CvtColor(image, ugray, ColorConversionCodes.BGRA2GRAY);
                    Cv2.EqualizeHist(ugray, ugray);
                    var facesDetected = face.DetectMultiScale(
                                image: ugray,
                                scaleFactor: 1.1,
                                minNeighbors: 10,
                                flags: HaarDetectionType.DoRoughSearch | HaarDetectionType.ScaleImage,
                                minSize: new OpenCvSharp.Size(20, 20));
                    faces.AddRange(facesDetected);
                }
            }
            return faces;
        }
    }


    class AccordFaceDetector
    {
        public static List<Rectangle> DetectFaces(Bitmap image)
        {
            List<Rectangle> xfaces = new List<Rectangle>();
            HaarObjectDetector detector;
            detector = new HaarObjectDetector(new FaceHaarCascade(), 20, ObjectDetectorSearchMode.Average, 1.1f, ObjectDetectorScalingMode.SmallerToGreater);
            detector.UseParallelProcessing = true;
            detector.Suppression = 2;
            var grayImage = Grayscale.CommonAlgorithms.BT709.Apply(image);
            HistogramEqualization filter = new HistogramEqualization();
            filter.ApplyInPlace(grayImage);
            Rectangle[] faces = detector.ProcessFrame(grayImage);
            xfaces.AddRange(faces);
            return xfaces;
        }
    }


    public class HomeController : Controller
    {
        public ActionResult Index()
        {
            if (Request.HttpMethod == "POST")
            {
                ViewBag.ImageProcessed = true;
                if (Request.Files.Count > 0)
                {
                    MemoryStream ms = new MemoryStream();
                    Request.Files[0].InputStream.CopyTo(ms);
                    var base64Data = Convert.ToBase64String(ms.ToArray()); //to display viewbag
                    var bitmap2 = new Bitmap(ms); //to face detect

                    var faces = EmguFaceDetector.DetectFaces(new Image<Bgr, byte>(bitmap2).Mat); //EMGUCV

                    if (faces.Count > 0)
                    {
                        ViewBag.FacesDetected = true;
                        ViewBag.FaceCount = faces.Count;

                        var positions = new List<Location>();
                        foreach (var face in faces)
                        {
                            positions.Add(new Location
                            {
                                X = face.X,
                                Y = face.Y,
                                Width = face.Width,
                                Height = face.Height
                            });
                        }

                        ViewBag.FacePositions = JsonConvert.SerializeObject(positions);
                    }

                    ViewBag.xImageUrl = base64Data;
                    ms.Dispose();
                }
            }

            return View();
        }


        public ActionResult About()
        {
            if (Request.HttpMethod == "POST")
            {
                ViewBag.ImageProcessed = true;
                if (Request.Files.Count > 0)
                {
                    MemoryStream ms = new MemoryStream();
                    Request.Files[0].InputStream.CopyTo(ms);
                    var base64Data = Convert.ToBase64String(ms.ToArray()); //to display viewbag
                    var bitmap2 = new Bitmap(ms); //to face detect

                    var faces = CvSharpFaceDetector.DetectFaces(BitmapConverter.ToMat(bitmap2)); //OPENCVSHARP

                    if (faces.Count > 0)
                    {
                        ViewBag.FacesDetected = true;
                        ViewBag.FaceCount = faces.Count;

                        var positions = new List<Location>();
                        foreach (var face in faces)
                        {
                            positions.Add(new Location
                            {
                                X = face.X,
                                Y = face.Y,
                                Width = face.Width,
                                Height = face.Height
                            });
                        }

                        ViewBag.FacePositions = JsonConvert.SerializeObject(positions);
                    }

                    ViewBag.xImageUrl = base64Data;
                    ms.Dispose();
                }
            }

            return View();
        }

        public ActionResult Contact()
        {
            if (Request.HttpMethod == "POST")
            {
                ViewBag.ImageProcessed = true;
                if (Request.Files.Count > 0)
                {
                    MemoryStream ms = new MemoryStream();
                    Request.Files[0].InputStream.CopyTo(ms);
                    var base64Data = Convert.ToBase64String(ms.ToArray()); //to display viewbag
                    var bitmap2 = new Bitmap(ms); //to face detect

                    var faces = AccordFaceDetector.DetectFaces(bitmap2); //ACCORD.NET

                    if (faces.Count > 0)
                    {
                        ViewBag.FacesDetected = true;
                        ViewBag.FaceCount = faces.Count;

                        var positions = new List<Location>();
                        foreach (var face in faces)
                        {
                            positions.Add(new Location
                            {
                                X = face.X,
                                Y = face.Y,
                                Width = face.Width,
                                Height = face.Height
                            });
                        }

                        ViewBag.FacePositions = JsonConvert.SerializeObject(positions);
                    }

                    ViewBag.xImageUrl = base64Data;
                    ms.Dispose();
                }
            }

            return View();
        }

    }
}