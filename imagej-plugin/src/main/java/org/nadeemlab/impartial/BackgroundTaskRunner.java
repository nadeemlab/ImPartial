package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;

public class BackgroundTaskRunner<Void> extends SwingWorker<Void, Void> {
    private final Callable<Void> task;
    private final Runnable onSuccess;
    private final Runnable onCancel;
    private final Consumer<Exception> onException;
    private final JDialog dialog;

    public BackgroundTaskRunner(Frame parent, String dialogMessage, Callable<Void> task, Runnable onSuccess,
                                Runnable onCancel, Consumer<Exception> onException) {
        this.task = task;
        this.onSuccess = onSuccess;
        this.onCancel = onCancel;
        this.onException = onException;

        dialog = createDialog(parent, dialogMessage);
    }

    public BackgroundTaskRunner(Frame parent, String dialogMessage, Callable<Void> task, Consumer<Exception> onException) {
        this(parent, dialogMessage, task, () -> {}, () -> {}, onException);
    }

    private JDialog createDialog(Frame parent, String dialogMessage) {
        JOptionPane optionPane = new JOptionPane(
                dialogMessage,
                JOptionPane.INFORMATION_MESSAGE,
                JOptionPane.DEFAULT_OPTION,
                null, new Object[]{}, null
        );
        optionPane.setOptions(null);

        JDialog dialog = new JDialog(parent, "Background task", false);
        dialog.setContentPane(optionPane);
        dialog.getRootPane().getDefaultButton().setVisible(false);
        dialog.setResizable(false);
        dialog.setDefaultCloseOperation(JDialog.DO_NOTHING_ON_CLOSE);
        dialog.setModal(false);
        dialog.setLocationRelativeTo(parent);
        dialog.pack();

        return dialog;
    }

    @Override
    protected Void doInBackground() throws Exception {
        dialog.setVisible(true);
        return task.call();
    }

    @Override
    protected void done() {
        Toolkit.getDefaultToolkit().beep();
        dialog.dispose();
        if (isCancelled()) {
            onCancel.run();
        } else {
            try {
                get();
                onSuccess.run();
            } catch (InterruptedException | ExecutionException e) {
                onException.accept(e);
            }
        }
    }
}