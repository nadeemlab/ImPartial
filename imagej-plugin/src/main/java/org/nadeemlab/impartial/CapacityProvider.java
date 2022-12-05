package org.nadeemlab.impartial;

import java.awt.*;
import javax.swing.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

public class CapacityProvider implements PropertyChangeListener {
    private final ImpartialController controller;
    private Task task;
    private ProgressMonitor progressMonitor;

    public CapacityProvider(ImpartialController controller) {
        this. controller = controller;
    }

    public void provisionServer() {
        progressMonitor = new ProgressMonitor(
                controller.getContentPane(), "request server...", "", 0, 100
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
            setProgress(30);
            try {
                controller.createSession();
                String status = controller.getSessionStatus();

                while (!status.equals("RUNNING") && !isCancelled()) {
                    Thread.sleep(1000);
                    status = controller.getSessionStatus();

                    if (status.equals("PROVISIONING")) setProgress(60);
                    else if (status.equals("PENDING")) setProgress(90);
                }

                Thread.sleep(20000);
                setProgress(100);

            } catch (InterruptedException ignore) {}
            return null;
        }

        @Override
        public void done() {
            Toolkit.getDefaultToolkit().beep();
            controller.onConnected();
        }
    }

    /**
     * Invoked when task's progress property changes.
     */
    public void propertyChange(PropertyChangeEvent e) {
        if (e.getPropertyName().equals("progress")) {
            int progress = (Integer) e.getNewValue();
            progressMonitor.setProgress(progress);
            String message;

            if (progress == 30) message = "generating token";
            else if (progress == 60) message = "provisioning server";
            else message = "task pending";

            progressMonitor.setNote(message);

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