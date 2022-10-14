package org.nadeemlab.impartial;

import ij.ImagePlus;
import ij.gui.*;
import ij.io.Opener;
import ij.plugin.frame.RoiManager;
import ij.process.FloatProcessor;
import net.imagej.ops.OpService;
import net.imagej.ops.geom.geom2d.LabelRegionToPolygonConverter;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.binary.Thresholder;
import net.imglib2.algorithm.labeling.ConnectedComponents;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.roi.geom.real.Polygon2D;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelRegion;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;
import org.json.JSONArray;
import org.json.JSONObject;
import org.scijava.Context;
import org.scijava.app.StatusService;
import org.scijava.plugin.Parameter;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.stream.StreamSupport;

public class ImpartialController {
    final JFileChooser fileChooser = new JFileChooser();
    private final JFrame mainFrame;
    private final ImpartialContentPane contentPane;
    private final MonaiLabelClient monaiClient = new MonaiLabelClient();
    private final Hashtable<String, ModelOutput> modelOutputs = new Hashtable<>();
    LabelRegionToPolygonConverter regionToPolygonConverter = new LabelRegionToPolygonConverter();
    private ImageWindow imageWindow;
    private File imageFile;
    private File labelFile;
    private int currentEpoch = 0;
    @Parameter
    private OpService ops;
    @Parameter
    private StatusService status;

    public ImpartialController(final Context context) {
        context.inject(this);
        context.inject(regionToPolygonConverter);
        createTempFiles();

        fileChooser.setMultiSelectionEnabled(true);

        mainFrame = new JFrame("ImPartial");
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainFrame.setResizable(false);

        //Create and set up the content pane.
        contentPane = new ImpartialContentPane(this);
        contentPane.setOpaque(true); //content panes must be opaque
        mainFrame.setContentPane(contentPane);

        mainFrame.pack();
        mainFrame.setVisible(true);
    }

    private void createTempFiles() {
        /*
         * This temporary files are used to store locally the current image,
         * and a zip file with rois for the label.
         * */
        try {
            imageFile = File.createTempFile("impartial-image-", ".png");
            imageFile.deleteOnExit();

            labelFile = File.createTempFile("impartial-label-", ".zip");
            labelFile.deleteOnExit();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void connect() {
        try {
            String[] samples = getDatastoreSamples();
            contentPane.populateSampleList(samples);
        } catch (IOException e) {
            JOptionPane.showMessageDialog(contentPane,
                    e.getMessage(),
                    e.getCause().getMessage(),
                    JOptionPane.ERROR_MESSAGE
            );
        }
    }

    public void updateImage(String imageId) {
        try {
            clearRoiManager();
            displayImage(imageId);

            JSONObject imageInfo = getImageInfo(imageId);
            boolean hasLabel = imageInfo.getJSONObject("labels").length() > 0;
            contentPane.setEnabledLabel(hasLabel);
            contentPane.setEnabledSubmit(!hasLabel);
            contentPane.setSelectedAll(false);

            contentPane.setEnabledInfer(true);
            if (modelOutputs.containsKey(imageId)) {
                ModelOutput output = modelOutputs.get(imageId);
                contentPane.setTextInfer(
                        "last run " + output.getTime() + ", epoch " + output.getEpoch()
                );

                contentPane.setEnabledInferAndEntropy(true);
            } else {
                contentPane.setTextInfer("last run never");
                contentPane.setEnabledInferAndEntropy(false);
            }

            updateDisplay();

        } catch (IOException e) {
            JOptionPane.showMessageDialog(contentPane,
                    e.getMessage(),
                    e.getCause().getMessage(),
                    JOptionPane.ERROR_MESSAGE
            );
        }
    }

    public void displayImage(String imageId) {
        retrieveImage(imageId);
        displayStoredImage();
    }

    private void retrieveImage(String imageId) {
        byte[] imageBytes = monaiClient.getDatastoreImage(imageId);

        try {
            FileOutputStream stream = new FileOutputStream(imageFile.getAbsolutePath());
            stream.write(imageBytes);
            stream.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void displayImage(ImagePlus imp) {
        if (imageWindow == null) {
            imageWindow = new ImageWindow(imp);
            resetLayout();
        } else {
            imageWindow.setImage(imp);
            imageWindow.pack();
        }

        if (!imageWindow.isVisible()) {
            imageWindow.setVisible(true);
            resetLayout();
        }
    }

    public void displayStoredImage() {
        final ImagePlus imp = new Opener().openImage(imageFile.getAbsolutePath());
        displayImage(imp);
    }

    public void setMonaiClientUrl(URL url) {
        monaiClient.setUrl(url);
    }

    public String[] getDatastoreSamples() throws IOException {
        JSONObject datastore = monaiClient.getDatastore();

        Iterable<String> iterable = () -> datastore.getJSONObject("objects").keys();
        return StreamSupport.stream(iterable.spliterator(), false)
                .toArray(String[]::new);
    }

    public void displayLabel() {
        try {
            String imageId = contentPane.getSelectedImageId();
            byte[] label = monaiClient.getDatastoreLabel(imageId);

            try {
                FileOutputStream stream = new FileOutputStream(labelFile);
                stream.write(label);
                stream.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            RoiManager roiManager = RoiManager.getRoiManager();

            JSONObject imageInfo = getImageInfo(imageId);
            if (imageInfo.getJSONObject("labels").length() > 0) {
                roiManager.runCommand("Open", labelFile.getAbsolutePath());
                roiManager.runCommand("Show All");
            }

        } catch (IOException e) {
            JOptionPane.showMessageDialog(contentPane,
                    e.getMessage(),
                    e.getCause().getMessage(),
                    JOptionPane.ERROR_MESSAGE
            );
        }
    }

    private void clearRoiManager() {
        Roi[] rois = RoiManager.getRoiManager().getRoisAsArray();
        if (rois.length > 0)
            RoiManager.getRoiManager().runCommand("Delete");
    }

    public void submitLabel() {
        if (RoiManager.getRoiManager().getCount() > 0) {
            String imageId = contentPane.getSelectedImageId();
            RoiManager.getRoiManager().runCommand("Save", labelFile.getAbsolutePath());
            monaiClient.putDatastoreLabel(imageId, labelFile.getAbsolutePath());
        } else {
            JOptionPane.showMessageDialog(contentPane,
                    "Please add at least one ROI using a selection tool",
                    "Empty ROI Manager",
                    JOptionPane.ERROR_MESSAGE
            );
        }
    }

    public JSONObject getImageInfo(String imageId) throws IOException {
        JSONObject datastore = monaiClient.getDatastore();
        return datastore.getJSONObject("objects").getJSONObject(imageId);
    }

    public void train() {
        monaiClient.deleteTrain();

        JSONObject params = contentPane.getTrainParams();
        monaiClient.postTrain("impartial", params);

        TrainProgress.monitorTraining(this);
    }

    public void infer() {
        ListModel listModel = contentPane.getListModel();

        for (int i = 0; i < listModel.size(); i++) {
            inferImage(listModel.getElementAt(i));
        }
    }

    private void inferImage(Sample sample) {
        SwingWorker<JSONObject, Void> swingWorker = new SwingWorker<JSONObject, Void>() {
            @Override
            protected JSONObject doInBackground() throws Exception {
                contentPane.setSampleStatus(sample, "running");

                String imageId = sample.getName();

                JSONObject params = new JSONObject();
                params.put("threshold", (Float) contentPane.getThreshold());

                JSONObject modelOutput = monaiClient.postInferJson("impartial", imageId, params);

                modelOutput.put("epoch", currentEpoch);

                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm");
                LocalDateTime time = LocalDateTime.now();
                modelOutput.put("time", time.format(formatter));

                return modelOutput;
            }

            @Override
            protected void done() {
                try {
                    JSONObject modelOutput = get();

                    String imageId = sample.getName();

                    DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm");
                    LocalDateTime time = LocalDateTime.now();

                    FloatProcessor output = jsonArrayToProcessor(modelOutput.getJSONArray("output"));
                    FloatProcessor entropy = jsonArrayToProcessor(modelOutput.getJSONArray("entropy"));

                    ModelOutput out = new ModelOutput(output, entropy, time.format(formatter), currentEpoch);

                    modelOutputs.put(imageId, out);

                    String selectedImageId = contentPane.getSelectedImageId();
                    if (selectedImageId.equals(imageId)) {
                        updateImage(imageId);
                    }

                    contentPane.setSampleStatus(sample, "done");
                    contentPane.setSampleEntropy(sample, entropy.getStatistics().mean);
                    contentPane.sortList();

                } catch (InterruptedException | ExecutionException e) {
                    throw new RuntimeException(e);
                }
            }
        };

        swingWorker.execute();
    }

    private void inferPerformed(String imageId) {
        ModelOutput modelOutput = modelOutputs.get(imageId);
        contentPane.inferPerformed(
                modelOutput.getEpoch(),
                modelOutput.getTime()
        );

        displayInfer();
    }

    public void displayInfer() {
        String imageId = contentPane.getSelectedImageId();
        FloatProcessor processor = modelOutputs.get(imageId).getOutput();
        ImagePlus imp = new ImagePlus("output", processor);

        Img<FloatType> img = ImageJFunctions.wrapFloat(imp);

        FloatType threshold = new FloatType(contentPane.getThreshold());
        Img<BitType> binaryImg = Thresholder.threshold(img, threshold, true, 2);

        final long[] dims = new long[binaryImg.numDimensions()];
        binaryImg.dimensions(dims);

        final RandomAccessibleInterval<UnsignedShortType> indexImg = ArrayImgs.unsignedShorts(dims);
        final ImgLabeling<Integer, UnsignedShortType> labeling = new ImgLabeling<>(indexImg);
        ops.labeling().cca(labeling, binaryImg, ConnectedComponents.StructuringElement.FOUR_CONNECTED);

        LabelRegions<Integer> regions = new LabelRegions<>(labeling);

        for (LabelRegion<Integer> region : regions) {
            Polygon2D contour = regionToPolygonConverter.convert(region, Polygon2D.class);

            int[] xs = contour.vertices().stream().mapToInt(p -> (int) p.getDoublePosition(0)).toArray();
            int[] ys = contour.vertices().stream().mapToInt(p -> (int) p.getDoublePosition(1)).toArray();

            PolygonRoi poly = new PolygonRoi(xs, ys, contour.numVertices(), Roi.POLYGON);

            RoiManager.getRoiManager().add(poly, region.getLabel());
        }
        RoiManager.getRoiManager().runCommand("Show All");
    }

    public void displayEntropy() {
        String imageId = contentPane.getSelectedImageId();
        FloatProcessor entropy = modelOutputs.get(imageId).getEntropy();

        ImageRoi roi = new ImageRoi(0, 0, entropy);
        roi.setZeroTransparent(true);

        Overlay overlay = new Overlay();
        overlay.add(roi);

        ImagePlus image = imageWindow.getImagePlus();
        image.setOverlay(overlay);
    }

    public FloatProcessor jsonArrayToProcessor(JSONArray input) {
        int width = input.length();
        int height = input.getJSONArray(0).length();

        FloatProcessor processor = new FloatProcessor(width, height);

        for (int i = 0; i < input.length(); i++) {
            JSONArray row = input.getJSONArray(i);
            for (int j = 0; j < row.length(); j++) {
                processor.setf(j, i, (float) row.getDouble(j));
            }
        }

        return processor;
    }

    public void showStatus(int progress, int max, String message) {
        status.showStatus(progress, max, message);
    }

    public JSONObject getInfo() {
        return monaiClient.getInfo();
    }

    public JSONObject getTrain() {
        return monaiClient.getTrain();
    }

    public int getMaxEpochs() {
        return contentPane.getTrainParams().getInt("max_epochs");
    }

    public void updateDisplay() {
        try {
            List<String> selected = contentPane.getSelectedViews();

            clearRoiManager();
            ImagePlus image = imageWindow.getImagePlus();
            image.setHideOverlay(true);

            if (selected.contains("entropy")) displayEntropy();
            if (selected.contains("label")) displayLabel();
            if (selected.contains("infer")) displayInfer();

            String imageId = contentPane.getSelectedImageId();
            boolean hasLabel = getImageInfo(imageId).getJSONObject("labels").length() > 0;
            contentPane.setEnabledSubmit(
                    !selected.contains("infer") && (!hasLabel || selected.contains("label"))
            );

            resetLayout();

        } catch (IOException e) {
            JOptionPane.showMessageDialog(contentPane,
                    e.getMessage(),
                    e.getCause().getMessage(),
                    JOptionPane.ERROR_MESSAGE
            );
        }
    }

    private void resetLayout() {
        Point mainFrameLocation = mainFrame.getLocation();
        imageWindow.setLocation(
                mainFrameLocation.x + mainFrame.getSize().width,
                mainFrameLocation.y
        );

        Point imageWindowLocation = imageWindow.getLocation();
        RoiManager.getRoiManager().setLocation(
                imageWindowLocation.x + imageWindow.getSize().width,
                imageWindowLocation.y
        );
    }

    public void uploadImages() {
        try {
            int returnVal = fileChooser.showOpenDialog(mainFrame);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                Arrays.stream(fileChooser.getSelectedFiles()).forEach(monaiClient::putDatastore);
            } else {
                System.out.println("Open command cancelled by user.");
            }

            String[] samples = getDatastoreSamples();
            contentPane.populateSampleList(samples);
        } catch (IOException e) {
            JOptionPane.showMessageDialog(contentPane,
                    e.getMessage(),
                    e.getCause().getMessage(),
                    JOptionPane.ERROR_MESSAGE
            );
        }
    }

    public void deleteSelectedImage() {
        String imageId = contentPane.getSelectedImageId();

        int option = JOptionPane.showConfirmDialog(
                contentPane,
                String.format("Are you sure you want to delete '%s' and corresponding labels?", imageId),
                "Delete confirmation",
                JOptionPane.YES_NO_OPTION);

        if (option == JOptionPane.YES_OPTION)
            try {
                monaiClient.deleteDatastore(imageId);
                String[] samples = getDatastoreSamples();
                contentPane.populateSampleList(samples);
                setSelectedFirstImage();
            } catch (IOException e) {
                JOptionPane.showMessageDialog(contentPane,
                        e.getMessage(),
                        e.getCause().getMessage(),
                        JOptionPane.ERROR_MESSAGE
                );
            }
    }

    private void setSelectedFirstImage() {
        contentPane.setSelectedFirstImage();
    }
}
