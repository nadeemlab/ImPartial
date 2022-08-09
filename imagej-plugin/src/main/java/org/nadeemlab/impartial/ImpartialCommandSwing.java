package org.nadeemlab.impartial;

import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import javax.swing.*;

@Plugin(type = Command.class, headless = true,
	menuPath = "Plugins>ImPartial")
public class ImpartialCommandSwing implements Command {

	@Parameter
	private Context context;
	@Parameter(type = ItemIO.OUTPUT)
	private static ImpartialDialog dialog = null;

	private void createAndShowGUI() {
		//Create and set up the window.
		JFrame frame = new JFrame("ImPartial");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		//Create and set up the content pane.
		ImpartialDialog impartial = new ImpartialDialog(context);
		impartial.mainPane.setOpaque(true); //content panes must be opaque
		frame.setContentPane(impartial.mainPane);

		//Display the window.
		frame.pack();
		frame.setVisible(true);
	}

	/**
	 * show a dialog and give the dialog access to required IJ2 Services
	 */
	@Override
	public void run() {
		SwingUtilities.invokeLater(() -> {
			createAndShowGUI();
//			if (dialog == null) {
//				dialog = new ImpartialDialog(context);
//			}
//			dialog.setVisible(true);
		});
	}
}
