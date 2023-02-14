package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;

public class ImageUploader implements PropertyChangeListener {
    private final ImpartialController controller;
    private File[] images;
    private Task task;
    private ProgressMonitor progressMonitor;

    public ImageUploader(ImpartialController controller) {
        this.controller = controller;
    }

    public void upload(File[] images) {
        this.images = images;

        progressMonitor = new ProgressMonitor(
                controller.getContentPane(), "uploading images...", "", 0, 100
        );

        progressMonitor.setProgress(0);
        progressMonitor.setMillisToDecideToPopup(0);
        progressMonitor.setMillisToPopup(0);

        task = new Task();
        task.addPropertyChangeListener(this);
        task.execute();
    }

    class Task extends SwingWorker<Void, Void> {
        @Override
        public Void doInBackground() {
            setProgress(0);
            int i = 0;
            while (i < images.length && !isCancelled()) {
                controller.uploadImage(images[i]);
                int progress = (int) (i++ * 100.0 / images.length);
                setProgress(progress);
            }
            setProgress(100);
            return null;
        }

        @Override
        public void done() {
            Toolkit.getDefaultToolkit().beep();
            controller.updateSampleList();
        }
    }

    public void propertyChange(PropertyChangeEvent e) {
        if (e.getPropertyName().equals("progress")) {
            int progress = (Integer) e.getNewValue();
            progressMonitor.setProgress(progress);

            if (progressMonitor.isCanceled() || task.isDone()) {
                Toolkit.getDefaultToolkit().beep();
                if (progressMonitor.isCanceled()) {
                    task.cancel(true);
                } else {
                    progressMonitor.close();
                }
            }
        }
    }
}