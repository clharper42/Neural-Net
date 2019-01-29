package neuralnet;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class Node
{
	boolean hiddenlayer;
	boolean outputnode;
	double input;
	double orginal_starting_input;
	double j_value;
	//int measured_output;
	int neuralnet_output;
	int calculated_output;
	ArrayList<Node> connections;
	ArrayList<Double> weights;
	Random r;
	
	public Node(double input, boolean hiddenlayer, boolean outputnode)
	{
		this.input = input;
		orginal_starting_input = input;
		//this.measured_output = measured_output;
		this.hiddenlayer = hiddenlayer;
		this.outputnode = outputnode;
		connections = new ArrayList<Node>();
		weights = new ArrayList<Double>();
		calculated_output = 0;
		r = new Random();
	}
	
	public void changeInput(double input)
	{
		this.input = input;
	}
	
	public double getInput()
	{
		return input;
	}
	
	public double getStartingInput()
	{
		return orginal_starting_input;
	}
	
	/*public int getMeasuredOutput()
	{
		return measured_output;
	}*/
	
	public boolean isaHiddenLayer()
	{
		return hiddenlayer;
	}
	
	public boolean isaOutputNode()
	{
		return outputnode;
	}
	
	
	/*public void addConnection(Node node)
	{
		connections.add(node);
	}
	*/
	public void addWeight(double w)
	{
		weights.add(w);
		//only hidden layer and output know weights (the weights are coming to them)
		//have a biased weight unconnected to input node
	}
	
	/*public ArrayList<Node> getConnectingNodes()
	{
		return connections;
	}
	*/
	
	public void addRandomWeight(int num_times)
	{
		double random;
		double upper = 0.1;
		double lower = -0.1;
		for(int i = 0; i < num_times; i++)
		{
			random = Math.random() * (upper - lower) + lower;
			weights.add(random);
		}
	}
	
	public ArrayList<Double> getWeights()
	{
		return weights;
	}
	
	public void updateWeight(double new_weight, int pos)
	{
		weights.set(pos,new_weight);
	}
	public void setJValue(double j_value)
	{
		this.j_value = j_value;
	}
	
	public double getJValue()
	{
		return j_value;
	}
}