using System;
using System.IO;
using System.Text;
using System.Linq;
using DeepLearning.Base;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace DeepLearning
{
    public class DeepLearningWrapper : IDisposable
    {
        #region Const

        public const int MAX_OBJECTS_NUMBER = 1000;
        //private const string DEEP_LEARNING_LIBRARY_DLL_CPU_VERSION = @"\x64\yolo_cpp_dll.dll";
        private const string DEEP_LEARNING_LIBRARY_DLL_GPU_VERSION = @"\x64\detector.dll";
        //private const string DEEP_LEARNING_LIBRARY_DLL_GPU_VERSION = @"\x64\yolo_cpp_dll.dll";

        #endregion

        #region Fields

        private static Dictionary<int, string> m_ObjectType = new Dictionary<int, string>();
        private static NetConfigaration m_NetConf;

        #endregion

        #region Initilizing

        public DeepLearningWrapper()
        {
        }

        #endregion
        
        #region DllImport Gpu

        [DllImport(DEEP_LEARNING_LIBRARY_DLL_GPU_VERSION, EntryPoint = "init")]
        internal static extern int InitializeNetInGpu(string configurationFilename, string weightsFilename);

        [DllImport(DEEP_LEARNING_LIBRARY_DLL_GPU_VERSION, EntryPoint = "detect_image")]
        internal static extern int DetectImageInGpu(string filename, ref BoundingBoxContainer container);

        [DllImport(DEEP_LEARNING_LIBRARY_DLL_GPU_VERSION, EntryPoint = "detect_mat")]
        internal static extern int DetectImageInGpu(IntPtr pArray, int nSize, ref BoundingBoxContainer container);

        [DllImport(DEEP_LEARNING_LIBRARY_DLL_GPU_VERSION, EntryPoint = "dispose")]
        internal static extern int DisposeNetInGpu();

        #endregion

        #region Dispose

        public void Dispose()
        {
            DisposeNetInGpu();
        }
        
        #endregion
        
        #region Wrapper

        public void InitilizeDeepLearningNetwork(int gpu = 0)
        {
            if (m_NetConf == null)
                m_NetConf = new NetConfigaration();
                                
            InitializeNetInGpu(m_NetConf.ConfigFile, m_NetConf.Weightfile);
            DetectionHardware = DetectionHardwares.GPU;

            var lines = File.ReadAllLines(m_NetConf.NamesFile);
            for (var i = 0; i < lines.Length; i++)
                m_ObjectType.Add(i, lines[i]);
        }


        public IEnumerable<DetectedItemInfo> Detect(string imageFilePath)
        {
            if (!File.Exists(imageFilePath))
                throw new FileNotFoundException("Cannot find the file", imageFilePath);

            var container = new BoundingBoxContainer();
            var count = 0;

            DetectImageInGpu(imageFilePath, ref container);
            return Convert(container);
        }

        
        public IEnumerable<DetectedItemInfo> Detect(byte[] imageData)
        {
            var container = new BoundingBoxContainer();

            var size = Marshal.SizeOf(imageData[0]) * imageData.Length;
            var pnt = Marshal.AllocHGlobal(size);

            try
            {
                // Copy the array to unmanaged memory.
                Marshal.Copy(imageData, 0, pnt, imageData.Length);
                var count = 0;
                
                count = DetectImageInGpu(pnt, imageData.Length, ref container);
                
            }
            catch (Exception exception)
            {
                return null;
            }
            finally
            {
                // Free the unmanaged memory.
                Marshal.FreeHGlobal(pnt);
            }

            return Convert(container);
        }
        
        #endregion

        #region Utilities

        private IEnumerable<DetectedItemInfo> Convert(BoundingBoxContainer container)
        {
            var netItems = new List<DetectedItemInfo>();
            foreach (var item in container.candidates.Where(o => o.height > 0 || o.width > 0))
            {
                var objectType = m_ObjectType[(int)item.obj_id];
                var netItem = new DetectedItemInfo() { X = (int)item.leftTopX, Y = (int)item.leftTopY, Height = (int)item.height, Width = (int)item.width, Confidence = item.prob, Type = objectType };
                netItems.Add(netItem);
            }

            return netItems;
        }

        #endregion

        #region Properties

        public DetectionHardwares DetectionHardware = DetectionHardwares.Unknown;
        public HardwareReports HardwareReport { get; private set; }

        #endregion
    }
}
