package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.IOException;

public class RestoreSessionTask implements PropertyChangeListener {
    private final ImpartialController controller;
    private JDialog dialog;
    private Task task;
    public RestoreSessionTask(ImpartialController controller) {
        this.controller = controller;
    }

    public void run() {
        JOptionPane optionPane = new JOptionPane(
                "Restoring session...",
                JOptionPane.INFORMATION_MESSAGE,
                JOptionPane.DEFAULT_OPTION,
                null, new Object[]{}, null
        );
        optionPane.setOptions(null);

        dialog = new JDialog(controller.getFrame(), "Background task", false);
        dialog.setContentPane(optionPane);
        dialog.getRootPane().getDefaultButton().setVisible(false);
        dialog.setResizable(false);
        dialog.setDefaultCloseOperation(JDialog.DO_NOTHING_ON_CLOSE);
        dialog.setModal(false);
        dialog.setLocationRelativeTo(controller.getFrame());
        dialog.pack();
        dialog.setVisible(true);

        task = new Task();
        task.addPropertyChangeListener(this);
        task.execute();
    }

    /**
     * Invoked when task's progress property changes.
     */
    public void propertyChange(PropertyChangeEvent e) {
        if (task.isDone() || task.isCancelled()) {
            dialog.setVisible(false);
            dialog.dispose();
        }
    }

    class Task extends SwingWorker<Void, Void> {
        @Override
        public Void doInBackground() {
            try {
                controller.restoreSession(controller.getSessionId());
                setProgress(100);
            } catch (IOException e) {
                this.cancel(true);
            }
            return null;
        }

        @Override
        public void done() {
            Toolkit.getDefaultToolkit().beep();
            if (isCancelled()) controller.onStopped();
            else controller.onStarted();
        }
    }
}